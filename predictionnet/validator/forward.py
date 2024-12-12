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

import time
from datetime import datetime, timedelta

import bittensor as bt
import pandas as pd
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
    """
    Process and decrypt data from a single miner response.

    Returns:
        Tuple[Optional[pd.DataFrame], Dict]: (processed DataFrame or None, predictions data)
    """
    processed_data = None
    if response.data and response.decryption_key and response.repo_id:
        # Construct full data path using repo_id
        full_data_path = f"{response.repo_id}/{response.data}"

        success, result = self.dataset_manager.decrypt_data(
            data_path=full_data_path, decryption_key=response.decryption_key
        )

        if success and isinstance(result.get("data"), pd.DataFrame):
            df = result["data"]
            # Add miner identification columns
            df["miner_uid"] = uid
            df["miner_hotkey"] = self.metagraph.hotkeys[uid]
            df["timestamp"] = timestamp
            processed_data = df
            bt.logging.success(f"Successfully decrypted data from UID {uid}")
        else:
            error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else "Invalid data format"
            bt.logging.error(f"Failed to decrypt data from UID {uid}: {error_msg}")

    predictions_data = {
        "prediction": response.prediction if response.prediction else None,
        "hotkey": self.metagraph.hotkeys[uid],
    }

    return processed_data, predictions_data, response.decryption_key


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

    # Collect DataFrames and predictions
    df_list = []
    predictions_data = {}
    encryption_keys = []

    for uid, response in zip(miner_uids, responses):
        proc_data, pred_data, encryption_key = await process_miner_response(self, uid, response, timestamp)
        if isinstance(proc_data, pd.DataFrame) and not proc_data.empty:
            df_list.append(proc_data)
            encryption_keys.append(encryption_key)
        predictions_data[str(uid)] = pred_data

    # Combine all valid DataFrame data
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)

        # Validate required columns
        required_columns = {"Open", "High", "Low", "Close", "Volume", "SMA_50", "SMA_200", "RSI", "CCI", "Momentum"}
        missing_columns = required_columns - set(combined_df.columns)
        if missing_columns:
            bt.logging.error(f"Combined data missing required columns: {missing_columns}")
            combined_df = pd.DataFrame()  # Reset to empty if validation fails
    else:
        bt.logging.warning("No valid data received from miners")
        combined_df = pd.DataFrame()

    # Add rewards to predictions data
    rewards = get_rewards(self, responses=responses, miner_uids=miner_uids)
    for uid, reward in zip(miner_uids, rewards.tolist()):
        predictions_data[str(uid)]["reward"] = float(reward)

    metadata = {
        "total_miners": len(miner_uids),
        "successful_decryptions": len(df_list),
        "market_conditions": {"timezone": ny_timezone.zone, "is_market_open": True},
        "encryption_keys": encryption_keys,  # Store the keys used for reference
    }

    await self.dataset_manager.store_data_async(
        timestamp=timestamp,
        miner_data=combined_df,
        predictions=predictions_data,
        encryption_key=encryption_keys[0] if encryption_keys else None,  # Use first valid key
        metadata=metadata,
    )

    models_confirmed = self.confirm_models(responses, miner_uids)
    bt.logging.info(f"Models Confirmed: {models_confirmed}")
    rewards = [0 if not model_confirmed else reward for reward, model_confirmed in zip(rewards, models_confirmed)]

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
    self.update_scores(rewards, miner_uids)
