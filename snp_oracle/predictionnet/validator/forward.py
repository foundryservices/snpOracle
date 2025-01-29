import asyncio
import time
from datetime import datetime, timedelta

import bittensor as bt
import wandb
from numpy import full, nan
from pytz import timezone

import snp_oracle.predictionnet as predictionnet
from snp_oracle.predictionnet.utils.dataset_manager import DatasetManager
from snp_oracle.predictionnet.utils.uids import check_uid_availability
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


async def process_miner_data(
    response, timestamp: str, organization: str, hotkey: str, uid: int, data_upload_on: bool = False
) -> bool:
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

        if not data_upload_on:
            bt.logging.info(f"Data upload disabled, skipping storage for UID {uid}")
            return True  # Return True since this isn't a failure case

        dataset_manager = DatasetManager(organization=organization)
        data_path = f"{response.repo_id}/{response.data}"

        # Verify decryption works
        success, result = dataset_manager.decrypt_data(data_path=data_path, decryption_key=response.decryption_key)

        if not success:
            bt.logging.error(f"Failed to decrypt data: {result['error']}")
            return False
        else:
            bt.logging.success(f"Successfully decrypted data for UID {uid} from {data_path}")

        # Store data in background without waiting for completion
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


async def handle_market_close(self, dataset_manager: DatasetManager, data_upload_on: bool) -> None:
    """Handle data management operations when market is closed."""
    try:
        if not data_upload_on:
            bt.logging.info("Data upload disabled, skipping market close operations")
            return

        # Clean up old data
        dataset_manager.cleanup_local_storage(days_to_keep=2)

        # Upload today's data
        success, result = dataset_manager.batch_upload_daily_data()
        if success:
            bt.logging.success(
                f"Daily batch upload completed. Uploaded {result.get('files_uploaded', 0)} files "
                f"with {result.get('total_rows', 0)} rows to {result.get('repo_id')}"
            )
        else:
            bt.logging.error(f"Daily batch upload failed: {result.get('error')}")

    except Exception as e:
        bt.logging.error(f"Error during market close operations: {str(e)}")


def log_to_wandb(wandb_on, miner_uids, responses, rewards):
    if wandb_on:
        try:
            wandb_val_log = {
                "miners_info": {
                    miner_uid: {
                        "miner_response": response.prediction,
                        "miner_reward": reward,
                    }
                    for miner_uid, response, reward in zip(miner_uids, responses, rewards.tolist())
                },
                "meta": {"prediction_timestamp": responses[0].timestamp},
            }

            bt.logging.debug(f"Attempting to log data to wandb: {wandb_val_log}")
            wandb.log(wandb_val_log)
        except Exception as e:
            bt.logging.error(f"Failed to log to wandb: {str(e)}")
            bt.logging.error("Full error: ", exc_info=True)


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It queries the network and scores responses when market is open,
    and handles data management when market is closed.
    """
    ny_timezone = timezone("America/New_York")
    current_time_ny = datetime.now(ny_timezone)
    daily_ops_done = False
    dataset_manager = None
    data_upload_on = getattr(self.config.neuron, "data_upload_on", False)

    if data_upload_on:
        dataset_manager = DatasetManager(organization=self.config.neuron.organization)

    while True:
        if await self.is_valid_time():
            bt.logging.info("Market is open. Begin processes requests")
            daily_ops_done = False  # Reset flag when market opens
            break
        else:
            bt.logging.info("Market is closed. Sleeping for 2 minutes...")
            time.sleep(120)  # Sleep for 5 minutes before checking again
            if datetime.now(ny_timezone) - current_time_ny >= timedelta(hours=1):
                self.resync_metagraph()
                self.set_weights()
                self.past_predictions = [full((self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan)] * len(self.hotkeys)
                current_time_ny = datetime.now(ny_timezone)

        if not daily_ops_done and data_upload_on and dataset_manager:
            await handle_market_close(self, dataset_manager, data_upload_on)
            daily_ops_done = True

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
    responses = await self.dendrite.forward(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=False,
        timeout=self.config.timeout,
    )

    # Process responses and track decryption success
    # Create tasks for all miners and wait for all decryption results
    decryption_tasks = []
    for uid, response in zip(miner_uids, responses):
        bt.logging.info(f"UID: {uid} | Predictions: {response.prediction}")

        if can_process_data(response):
            task = process_miner_data(
                response=response,
                timestamp=timestamp,
                organization=self.config.neuron.organization,
                hotkey=self.metagraph.hotkeys[uid],
                uid=uid,
                data_upload_on=data_upload_on,
            )
            decryption_tasks.append(task)
        else:
            decryption_tasks.append(asyncio.create_task(asyncio.sleep(0)))  # Dummy task that returns immediately

    # Wait for all decryption results
    # decryption_success = await asyncio.gather(*decryption_tasks)

    # Calculate initial rewards
    rewards = get_rewards(self, responses=responses, miner_uids=miner_uids)

    # Zero out rewards for failed decryption
    # rewards = [reward if success else 0 for reward, success in zip(rewards, decryption_success)]

    wandb_on = self.config.neuron.wandb_on
    log_to_wandb(wandb_on, miner_uids, responses, rewards)

    # Log scores and update
    bt.logging.info(f"Scored responses: {rewards}")
    models_confirmed = self.confirm_models(responses, miner_uids)
    bt.logging.info(f"Models Confirmed: {models_confirmed}")
    rewards = [0 if not model_confirmed else reward for reward, model_confirmed in zip(rewards, models_confirmed)]
    self.update_scores(rewards, miner_uids)
