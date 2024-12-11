# The MIT License (MIT)
# Copyright © 2024 Foundry Digital

import time
from datetime import datetime, timedelta

import bittensor as bt
import wandb
from numpy import full, nan
from pytz import timezone

import predictionnet
from predictionnet.utils.dataset_manager import DatasetManager
from predictionnet.utils.uids import check_uid_availability
from predictionnet.validator.reward import get_rewards


async def get_available_miner_uids(self):
    """Get list of available miner UIDs."""
    miner_uids = []
    for uid in range(len(self.metagraph.S)):
        uid_is_available = check_uid_availability(self.metagraph, uid, self.config.neuron.vpermit_tao_limit)
        if uid_is_available:
            miner_uids.append(uid)
    return miner_uids


async def process_miner_response(self, uid, response, timestamp):
    """Process and decrypt data from a single miner response."""
    processed_data = None
    if response.data and response.decryption_key and response.repo_id:
        # Construct full data path using repo_id
        full_data_path = f"{response.repo_id}/{response.data}"

        success, data = self.dataset_manager.decrypt_data(
            data_path=full_data_path, decryption_key=response.decryption_key
        )

        if success:
            processed_data = {"data": data, "hotkey": self.metagraph.hotkeys[uid], "timestamp": timestamp}
            bt.logging.success(f"Successfully decrypted data from UID {uid}")
        else:
            bt.logging.error(f"Failed to decrypt data from UID {uid}: {data['error']}")

    predictions_data = {
        "prediction": response.prediction if response.prediction else None,
        "hotkey": self.metagraph.hotkeys[uid],
    }

    return processed_data, predictions_data


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    """
    # Initialize dataset manager if not already done
    if not hasattr(self, "dataset_manager"):
        self.dataset_manager = DatasetManager(organization=self.config.neuron.organization)

    ny_timezone = timezone("America/New_York")
    current_time_ny = datetime.now(ny_timezone)
    bt.logging.info("Current time: ", current_time_ny)

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

    miner_uids = await get_available_miner_uids(self)
    timestamp = datetime.now(ny_timezone).isoformat()

    synapse = predictionnet.protocol.Challenge(timestamp=timestamp)
    responses = self.dendrite.query(
        axons=[self.metagraph.axons[uid] for uid in miner_uids],
        synapse=synapse,
        deserialize=False,
    )

    processed_data = {}
    predictions_data = {}

    for uid, response in zip(miner_uids, responses):
        proc_data, pred_data = await process_miner_response(self, uid, response, timestamp)
        if proc_data:
            processed_data[str(uid)] = proc_data
        predictions_data[str(uid)] = pred_data

    rewards = get_rewards(self, responses=responses, miner_uids=miner_uids)
    for uid, reward in zip(miner_uids, rewards.tolist()):
        predictions_data[str(uid)]["reward"] = float(reward)

    metadata = {
        "total_miners": len(miner_uids),
        "successful_decryptions": sum(
            1 for uid_data in processed_data.values() if "error" not in uid_data.get("data", {})
        ),
        "market_conditions": {"timezone": ny_timezone.zone, "is_market_open": True},
    }

    await self.dataset_manager.store_data_async(
        timestamp=timestamp, miner_data=processed_data, predictions=predictions_data, metadata=metadata
    )

    wandb_val_log = {
        "miners_info": {
            miner_uid: {
                "miner_response": response.prediction,
                "miner_reward": reward,
                "data_decrypted": str(miner_uid) in processed_data,
            }
            for miner_uid, response, reward in zip(miner_uids, responses, rewards.tolist())
        }
    }
    wandb.log(wandb_val_log)

    models_confirmed = self.confirm_models(responses, miner_uids)
    bt.logging.info(f"Models Confirmed: {models_confirmed}")
    rewards = [0 if not model_confirmed else reward for reward, model_confirmed in zip(rewards, models_confirmed)]
    self.update_scores(rewards, miner_uids)
