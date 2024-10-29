import os
from datetime import datetime

import bittensor as bt
import psycopg
from dotenv import load_dotenv
from psycopg import OperationalError, sql

from predictionnet.api.get_query_axons import get_query_api_axons
from predictionnet.api.prediction import PredictionAPI

bt.debug()
load_dotenv()
pg_connection_string = os.environ.get("POSTGRESQL_CONNECTION_STRING")

# ---------------------------------------------------------------------------- #
#                             Select/Insert/Update                             #
# ---------------------------------------------------------------------------- #
find_miner_by_hot_key_cold_key = sql.SQL(
    "SELECT * FROM miners_table WHERE hot_key = %s AND cold_key = %s"
)
update_miner_by_hot_key_cold_key = sql.SQL(
    "UPDATE miners_table SET uid = %s, is_current_uid = %s, rank = %s, trust = %s WHERE hot_key = %s AND cold_key = %s"
)
update_miner_uid_to_false_by_hot_key_cold_key = sql.SQL(
    "UPDATE miners_table SET is_current_uid = %s WHERE NOT hot_key = %s AND NOT cold_key = %s AND uid=%s"
)
insert_miner_query = sql.SQL(
    "INSERT INTO miners_table (hot_key, cold_key, uid, is_current_uid, rank, trust) VALUES (%s, %s, %s, %s, %s, %s)"
)
insert_prediction_query = sql.SQL(
    "INSERT INTO predictions_table (prediction, timestamp, miner_id) VALUES (%s, %s, %s)"
)


# ---------------------------------------------------------------------------- #
#                                 Connect to DB                                #
# ---------------------------------------------------------------------------- #
def create_connection(conn_str):
    """
    Create a database connection using the provided connection string.
    :param conn_str: Database connection string
    :return: Connection object or None
    """
    conn = None
    try:
        conn = psycopg.connect(conn_str)
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return conn


# ---------------------------------------------------------------------------- #
#                                 Example usage                                #
# ---------------------------------------------------------------------------- #
async def test_prediction():
    wallet = bt.wallet()

    # Fetch the axons of the available API nodes, or specify UIDs directly
    metagraph = bt.subtensor("local").metagraph(netuid=28)

    uids = [uid.item() for uid in metagraph.uids if metagraph.trust[uid] > 0]

    axons = await get_query_api_axons(
        wallet=wallet, metagraph=metagraph, uids=uids
    )

    # Store some data!
    # Read timestamp from the text file
    with open("timestamp.txt", "r") as file:
        timestamp = file.read()

    bt.logging.info(f"Sending {timestamp} to predict a price.")
    retrieve_handler = PredictionAPI(wallet)
    retrieve_response = await retrieve_handler(
        axons=axons,
        # Arugmnts for the proper synapse
        timestamp=timestamp,
        timeout=120,
    )

    ranks = metagraph.R
    ck = metagraph.coldkeys
    hk = metagraph.hotkeys
    trust = metagraph.T

    exporter_list = []

    # For each UID, store the predictions in PostgresQL:
    connection = create_connection(pg_connection_string)
    cursor = connection.cursor()
    for i in range(len(retrieve_response)):
        export_dict = {}
        export_dict["UID"] = str(uids[i])
        export_dict["prediction"] = retrieve_response[i]
        export_dict["rank"] = ranks[uids[i]].item()
        export_dict["trust"] = trust[uids[i]].item()
        export_dict["hotKey"] = hk[uids[i]]
        export_dict["coldKey"] = ck[uids[i]]
        print(export_dict)
        exporter_list.append(export_dict)
        # -------------------- Insert/Update the miner in the DB: -------------------- #
        # Check if HKey + CKey exists in the DB
        # This returns an array with all the matching rows. You can use array length to check if a miner was found.
        _ = cursor.execute(
            find_miner_by_hot_key_cold_key,
            (export_dict["hotKey"], export_dict["coldKey"]),
        )
        result = cursor.fetchall()
        if len(result) == 0:
            # Miner not found, insert miner:
            _ = cursor.execute(
                insert_miner_query,
                (
                    export_dict["hotKey"],
                    export_dict["coldKey"],
                    export_dict["UID"],
                    True,
                    export_dict["rank"],
                    export_dict["trust"],
                ),
            )
            connection.commit()
        else:
            # Miner found, update miner:
            _ = cursor.execute(
                update_miner_by_hot_key_cold_key,
                (
                    export_dict["UID"],
                    True,
                    export_dict["rank"],
                    export_dict["trust"],
                    export_dict["hotKey"],
                    export_dict["coldKey"],
                ),
            )
            connection.commit()

        # Update the other UID that is not the same HK + CK to False:
        _ = cursor.execute(
            update_miner_uid_to_false_by_hot_key_cold_key,
            (
                False,
                export_dict["hotKey"],
                export_dict["coldKey"],
                export_dict["UID"],
            ),
        )
        connection.commit()

        # ------------------------ Fetch the updated version: ------------------------ #
        _ = cursor.execute(
            find_miner_by_hot_key_cold_key,
            (export_dict["hotKey"], export_dict["coldKey"]),
        )
        result = cursor.fetchall()
        updated_miner = result[0]

        # Insert the prediction:
        cursor.execute(
            insert_prediction_query,
            (
                export_dict["prediction"],
                datetime.datetime.now().isoformat(),
                updated_miner[0],
            ),
        )
        connection.commit()


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_prediction())
