import asyncio
import time
from datetime import datetime, timedelta

import bittensor as bt
import wandb
from numpy import full, nan
from pytz import timezone

from snp_oracle.predictionnet.protocol import Challenge
from snp_oracle.predictionnet.utils.timestamp import datetime_to_iso8601, get_now

import snp_oracle.predictionnet as predictionnet
from snp_oracle.predictionnet.utils.dataset_manager import DatasetManager
from snp_oracle.predictionnet.utils.bittensor import check_uid_availability
from snp_oracle.predictionnet.validator.reward import get_rewards


def can_process_data(response) -> bool:
    """
    Check if a response has required data fields populated.

    Args:
        response: Miner response object

    Returns:
        bool: True if all required fields have non-empty values
    """
    return all([bool(response.repo_id), bool(response.data), bool(response.decryption_key), bool(response.prediction)])


async def process_miner_data(self, response, timestamp: str, organization: str, hotkey: str, uid: int) -> bool:
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
        data_path = f"{response.repo_id}/{response.data}"

        # Verify decryption works
        success, result = self.dataset_manager.decrypt_data(data_path=data_path, decryption_key=response.decryption_key)

        if not success:
            bt.logging.error(f"Failed to decrypt data: {result['error']}")
            return False
        else:
            bt.logging.success(f"Successfully decrypted data for UID {uid} from {data_path}")

        # Store data in background without waiting for completion
        asyncio.create_task(
            self.dataset_manager.store_data_async(
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
    It queries the network and scores responses when market is open,
    and handles data management when market is closed.
    """
    # Get current timestamp
    timestamp = datetime_to_iso8601(get_now())
    # Build synapse for request
    synapse = Challenge(timestamp=timestamp)

    # Query miners
    responses = self.dendrite.query(
        axons=[self.metagraph.axons[uid] for uid in self.available_uids],
        synapse=synapse,
        deserialize=False,
    )

    # Process responses and track decryption success
    # Create tasks for all miners and wait for all decryption results
    decryption_tasks = []
    for uid, response in zip(self.available_uids, responses):
        bt.logging.info(f"UID: {uid} | Predictions: {response.prediction}")

        if can_process_data(response):
            task = process_miner_data(
                response=response,
                timestamp=timestamp,
                organization=self.config.neuron.organization,
                hotkey=self.metagraph.hotkeys[uid],
                uid=uid,
            )
            decryption_tasks.append(task)
        else:
            decryption_tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task that returns immediately

    # Wait for all decryption results
    decryption_success = await asyncio.gather(*decryption_tasks)

    # Calculate initial rewards
    rewards = get_rewards(self, responses=responses)

    # Zero out rewards for failed decryption
    rewards = [reward if success else 0 for reward, success in zip(rewards, decryption_success)]

    # Log results to wandb
    wandb_val_log = {
        "miners_info": {
            miner_uid: {"miner_response": response.prediction, "miner_reward": reward, "decryption_success": success}
            for miner_uid, response, reward, success in zip(self.available_uids, responses, rewards, decryption_success)
        }
    }
    wandb.log(wandb_val_log)

    # Log scores and update
    bt.logging.info(f"Scored responses: {rewards}")
    models_confirmed = self.confirm_models(responses)
    bt.logging.info(f"Models Confirmed: {models_confirmed}")
    rewards = [0 if not model_confirmed else reward for reward, model_confirmed in zip(rewards, models_confirmed)]
    self.update_scores(rewards)
