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
import asyncio
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


def can_process_data(response) -> bool:
    """
    Check if a response has required data fields populated.

    Args:
        response: Miner response object

    Returns:
        bool: True if all required fields have non-empty values
    """
    return all([bool(response.repo_id), bool(response.data), bool(response.decryption_key), bool(response.prediction)])


async def process_miner_data(response, timestamp: str, organization: str, hotkey: str, uid: int) -> bool:
    """
    Verify that miner's data can be decrypted and attempt to store it.

    Args:
        response: Response from miner containing encrypted data
        timestamp: Current timestamp
        organization: Organization name for HuggingFace
        hotkey: Miner's hotkey for data organization
        uid: Miner's UID

    Returns:
        bool: True if data was successfully decrypted, False otherwise
    """
    try:
        bt.logging.info(f"Processing data from UID {uid}...")
        dataset_manager = DatasetManager(organization=organization)
        data_path = f"{response.repo_id}/{response.data}"

        # Verify decryption works
        success, result = dataset_manager.decrypt_data(data_path=data_path, decryption_key=response.decryption_key)

        if not success:
            bt.logging.error(f"Failed to decrypt data: {result['error']}")
            return False

        # Attempt to store data in background without affecting reward
        asyncio.create_task(
            dataset_manager.store_data_async(
                timestamp=timestamp,
                miner_data=result["data"],
                predictions=result.get("predictions", {}),
                hotkey=hotkey,
                metadata={"source_uid": str(uid), "original_repo": response.repo_id, **result.get("metadata", {})},
            )
        )

        return True

    except Exception as e:
        bt.logging.error(f"Error processing data from UID {uid}: {str(e)}")
        return False


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
    """
    # Market timing setup
    ny_timezone = timezone("America/New_York")
    current_time_ny = datetime.now(ny_timezone)
    bt.logging.info("Current time: ", current_time_ny)

    # Block forward from running if market is closed
    while True:
        if await self.is_valid_time():
            bt.logging.info("Market is open. Begin processes requests")
            break
        else:
            bt.logging.info("Market is closed. Sleeping for 2 minutes...")
            time.sleep(120)
            if datetime.now(ny_timezone) - current_time_ny >= timedelta(hours=1):
                self.resync_metagraph()
                self.set_weights()
                self.past_predictions = [full((self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan)] * len(self.hotkeys)
                current_time_ny = datetime.now(ny_timezone)

    # Get available miner UIDs
    miner_uids = []
    for uid in range(len(self.metagraph.S)):
        uid_is_available = check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        if uid_is_available:
            miner_uids.append(uid)

    # Get current timestamp
    current_time_ny = datetime.now(ny_timezone)
    timestamp = current_time_ny.isoformat()

    # Build synapse for request
    synapse = predictionnet.protocol.Challenge(timestamp=timestamp)

    # Query miners
    responses = self.dendrite.query(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=False,
    )

    # Process responses and initiate background data processing
    for uid, response in zip(miner_uids, responses):
        bt.logging.info(f"UID: {uid} | Predictions: {response.prediction}")

        # Check if response has required data fields
        if can_process_data(response):
            # Create background task for data processing
            asyncio.create_task(
                process_miner_data(
                    response=response,
                    timestamp=timestamp,
                    organization=self.config.neuron.organization,
                    hotkey=self.metagraph.hotkeys[uid],
                    uid=uid,
                )
            )

    # Calculate rewards
    rewards = get_rewards(self, responses=responses, miner_uids=miner_uids)

    # Log results to wandb
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

    # Log scores and update
    bt.logging.info(f"Scored responses: {rewards}")
    models_confirmed = self.confirm_models(responses, miner_uids)
    bt.logging.info(f"Models Confirmed: {models_confirmed}")
    rewards = [0 if not model_confirmed else reward for reward, model_confirmed in zip(rewards, models_confirmed)]
    self.update_scores(rewards, miner_uids)
