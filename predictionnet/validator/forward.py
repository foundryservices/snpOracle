# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import os
import time
from datetime import datetime, timedelta

import bittensor as bt
import wandb
from numpy import full, nan
from pytz import timezone

# Import Validator Template
import predictionnet
from predictionnet.utils.dataset_manager import DatasetManager
from predictionnet.utils.uids import check_uid_availability
from predictionnet.validator.reward import get_rewards


def process_uid_146_data(response, timestamp: str, organization: str):
    """
    Decrypt and store unencrypted data from UID 146 in the organization dataset.

    Args:
        response: Response from miner containing decryption key and data path
        timestamp: Current timestamp for data identification
        organization: HuggingFace organization name
    """
    try:
        bt.logging.info("Processing data from UID 146...")

        # Initialize DatasetManager with explicit organization
        dataset_manager = DatasetManager(organization=organization)

        bt.logging.info(response)

        combined_path = response.repo_id + response.data

        bt.logging.info(f"Attempting to decrypt data from path: {combined_path}")

        # Attempt to decrypt the data
        success, result = dataset_manager.decrypt_data(data_path=combined_path, decryption_key=response.decryption_key)

        if not success:
            bt.logging.error(f"Failed to decrypt data: {result['error']}")
            return

        # Get the decrypted data
        df = result["data"]

        bt.logging.info(f"Successfully decrypted data with shape: {df.shape}")

        # Get current repo name based on date
        repo_name = f"dataset-{datetime.now().strftime('%Y-%m')}"
        repo_id = f"{organization}/{repo_name}"

        try:
            # Save as regular CSV
            filename = f"market_data_{timestamp}.csv"
            df.to_csv(filename, index=True)

            # Upload to HuggingFace
            dataset_manager.api.upload_file(
                path_or_fileobj=filename, path_in_repo=filename, repo_id=repo_id, create_pr=False
            )

            # Clean up local file
            os.remove(filename)

            bt.logging.success(f"Successfully uploaded unencrypted data to {repo_id}/{filename}")

        except Exception as e:
            bt.logging.error(f"Failed to upload data: {str(e)}")

    except Exception as e:
        bt.logging.error(f"Error processing UID 146 data: {str(e)}")
        bt.logging.error(f"Response data path: {response.data}")
        bt.logging.error(f"Response repo ID: {response.repo_id}")


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.

    # wait for market to be open
    ny_timezone = timezone("America/New_York")
    current_time_ny = datetime.now(ny_timezone)
    bt.logging.info("Current time: ", current_time_ny)
    # block forward from running if market is closed
    while True:
        if await self.is_valid_time():
            bt.logging.info("Market is open. Begin processes requests")
            break
        else:
            bt.logging.info("Market is closed. Sleeping for 2 minutes...")
            time.sleep(120)  # Sleep for 5 minutes before checking again
            if datetime.now(ny_timezone) - current_time_ny >= timedelta(hours=1):
                self.resync_metagraph()
                self.set_weights()
                self.past_predictions = [full((self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan)] * len(self.hotkeys)
                current_time_ny = datetime.now(ny_timezone)

    # miner_uids = get_random_uids(self, k=min(self.config.neuron.sample_size, self.metagraph.n.item()))
    # get all uids
    miner_uids = []
    for uid in range(len(self.metagraph.S)):
        uid_is_available = check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        if uid_is_available:
            miner_uids.append(uid)

    # Here input data should be gathered to send to the miners
    # TODO(create get_input_data())
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()

    # Build synapse for request
    # Replace dummy_input with actually defined variables in protocol.py
    # This can be combined with line 49
    synapse = predictionnet.protocol.Challenge(
        timestamp=timestamp,
    )

    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        # Construct a dummy query. This simply contains a single integer.
        # This can be simplified later to all build from here
        synapse=synapse,
        # synapse=Dummy(dummy_input=self.step),
        # All responses have the deserialize function called on them before returning.
        # You are encouraged to define your own deserialization function.
        # Other subnets have this turned to false, I am unsure of whether this should be set to true
        deserialize=False,
    )
    # Log the results for monitoring purposes.
    for uid, response in zip(miner_uids, responses):
        bt.logging.info(f"UID: {uid} | Predictions: {response.prediction}")

        if uid == 146:
            bt.logging.info("Processing special case for UID 146...")
            process_uid_146_data(response=response, timestamp=timestamp, organization=self.config.neuron.organization)

    rewards = get_rewards(self, responses=responses, miner_uids=miner_uids)

    wandb_val_log = {
        "miners_info": {
            miner_uid: {
                "miner_response": response.prediction,
                "miner_reward": reward,
            }
            for miner_uid, response, reward in zip(miner_uids, responses, rewards.tolist())
        }
    }

    wandb.log(wandb_val_log)

    # Potentially will need some
    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.

    # Check base validator file
    self.update_scores(rewards, miner_uids)
