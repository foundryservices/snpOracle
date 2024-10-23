import bittensor as bt

from predictionnet.api.get_query_axons import get_query_api_axons
from predictionnet.api.prediction import PredictionAPI

bt.debug()


# Example usage
async def test_prediction():
    wallet = bt.wallet()

    # Fetch the axons of the available API nodes, or specify UIDs directly
    metagraph = bt.subtensor("finney").metagraph(netuid=28)

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
    print(retrieve_response)
    print(uids)
    print(len(retrieve_response))
    print(len(uids))


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_prediction())
