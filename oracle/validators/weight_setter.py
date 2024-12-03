import asyncio
import os
import pickle

import bittensor as bt
from aiohttp import ClientConnectorError
from numpy import array, isnan
from substrateinterface import SubstrateInterface

from oracle import __spec_version__
from oracle.protocol import Challenge
from oracle.utils.bittensor import get_available_uids, node_query, print_info, setup_bittensor_objects
from oracle.utils.classes import MinerHistory
from oracle.utils.general import loop_handler
from oracle.utils.timestamp import (
    elapsed_seconds,
    get_before,
    get_now,
    is_query_time,
    iso8601_to_datetime,
    market_is_open,
)
from oracle.utils.wandb import log_wandb, setup_wandb
from oracle.validators.reward import calc_rewards


class weight_setter:
    def __init__(self, config=None, loop=None):
        self.config = config
        self.loop = loop
        self.lock = asyncio.Lock()
        setup_bittensor_objects(self)
        self.prediction_interval = self.config.prediction_interval  # in minutes
        self.N_TIMEPOINTS = self.config.N_TIMEPOINTS  # number of timepoints to predict
        self.last_sync = 0
        self.set_weights_rate = 100  # in blocks
        self.resync_metagraph_rate = 40  # in blocks
        bt.logging.info(
            f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network}"
        )
        self.available_uids = asyncio.run(get_available_uids(self))
        self.hotkeys = self.metagraph.hotkeys
        if self.config.reset_state:
            self.scores = [0.0] * len(self.metagraph.S)
            self.moving_average_scores = {uid: 0 for uid in self.metagraph.uids}
            self.MinerHistory = {uid: MinerHistory(uid) for uid in self.available_uids}
            self.timestamp = get_before(minutes=60)
            self.save_state()
        else:
            self.load_state()
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        self.current_block = self.subtensor.get_current_block()
        self.blocks_since_last_update = (
            self.current_block - node_query(self, "SubtensorModule", "LastUpdate", [self.config.netuid])[self.my_uid]
        )
        self.tempo = node_query(self, "SubtensorModule", "Tempo", [self.config.netuid])
        if self.config.wandb_on:
            setup_wandb(self)
        self.stop_event = asyncio.Event()
        bt.logging.info("Setup complete, starting loop")
        self.loop.create_task(loop_handler(self, self.main_function, sleep_time=self.config.print_cadence))
        self.loop.create_task(loop_handler(self, self.resync_metagraph, sleep_time=self.resync_metagraph_rate))
        self.loop.create_task(loop_handler(self, self.set_weights, sleep_time=self.set_weights_rate))

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_state()
        try:
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            asyncio.gather(*pending)
        except Exception as e:
            bt.logging.error(f"Error on __exit__ function: {e}")
        self.loop.stop()

    async def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        try:
            self.blocks_since_sync = self.current_block - self.last_sync
            if self.blocks_since_sync >= self.resync_metagraph_rate:
                bt.logging.info("Syncing Metagraph...")
                self.metagraph.sync(subtensor=self.subtensor)
                bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
                # Zero out all hotkeys that have been replaced.
                self.available_uids = asyncio.run(get_available_uids(self))
                for uid, hotkey in enumerate(self.metagraph.hotkeys):
                    if (uid not in self.MinerHistory and uid in self.available_uids) or self.hotkeys[uid] != hotkey:
                        bt.logging.info(f"Replacing hotkey on {uid} with {self.metagraph.hotkeys[uid]}")
                        self.hotkeys[uid] = hotkey
                        self.MinerHistory[uid] = MinerHistory(uid)
                        self.moving_average_scores[uid] = 0
                        self.scores = array(list(self.moving_average_scores.values()))
                self.last_sync = self.subtensor.get_current_block()
                self.save_state()
        except Exception as e:
            bt.logging.error(f"Resync metagraph error: {e}")
            raise e

    def query_miners(self):
        bt.logging.info("Querying miners...")
        timestamp = get_now().isoformat()
        synapse = Challenge(timestamp=timestamp)
        connect_issues = 0
        Timeouts = 0
        try:
            responses = self.dendrite.query(
                # Send the query to selected miner axons in the network.
                axons=[self.metagraph.axons[uid] for uid in self.available_uids],
                synapse=synapse,
                deserialize=False,
            )
        except ClientConnectorError:
            connect_issues += 1
        except TimeoutError:
            Timeouts += 1
        finally:
            bt.logging.debug(f"Connect issues: {connect_issues} | Timeouts: {Timeouts}")
        return responses

    async def set_weights(self):
        try:
            if self.blocks_since_last_update >= self.set_weights_rate:
                uids = array(self.available_uids)
                weights = [self.moving_average_scores[uid] for uid in self.available_uids]
                if isnan(weights).any():
                    bt.logging.error("Weights contain all NaN values. Setting weights to 0.")
                    weights = [0] * len(weights)
                for i, j in zip(weights, self.available_uids):
                    bt.logging.debug(f"UID: {j}  |  Weight: {i}")
                if sum(weights) == 0:
                    weights = [1] * len(weights)
                weights = array(weights) / max(weights)
                # Convert to uint16 weights and uids.
                (
                    uint_uids,
                    uint_weights,
                ) = bt.utils.weight_utils.convert_weights_and_uids_for_emit(uids=uids, weights=weights)
                # Update the incentive mechanism on the Bittensor blockchain.
                result, msg = self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=uint_uids,
                    weights=uint_weights,
                    wait_for_inclusion=False,
                    version_key=__spec_version__,
                )
                if result:
                    bt.logging.success("✅ Set Weights on chain successfully!")
                else:
                    bt.logging.debug(
                        "Failed to set weights this iteration with message:",
                        msg,
                    )
            self.current_block = node_query(self, "System", "Number", [])
            self.blocks_since_last_update = (
                self.current_block
                - node_query(self, "SubtensorModule", "LastUpdate", [self.config.netuid])[self.my_uid]
            )
        except Exception as e:
            bt.logging.error(f"set_weights loop error: {e}")
            raise e

    async def main_function(self):
        if market_is_open():
            query_lag = elapsed_seconds(get_now(), self.timestamp)
            if is_query_time(self.prediction_interval, self.timestamp) or query_lag >= 60 * self.prediction_interval:
                responses = self.query_miners()
                self.timestamp = iso8601_to_datetime(responses[0].timestamp)
                try:
                    rewards = calc_rewards(
                        self,
                        responses=responses,
                    )
                except Exception as e:
                    bt.logging.error(f"Error calculating rewards: {e}")
                    raise e
                for uid, response, reward in zip(self.available_uids, responses, rewards):
                    if response.prediction is not None:
                        bt.logging.info(f"UID: {uid}  |  Prediction: {response.prediction}  |  Reward: {reward}")
                # Adjust the scores based on responses from miners and update moving average.
                for i, value in zip(self.available_uids, rewards):
                    self.moving_average_scores[i] = (1 - self.config.alpha) * self.moving_average_scores[
                        i
                    ] + self.config.alpha * value
                self.scores = array(list(self.moving_average_scores.values()))
                bt.logging.debug(f"Scores: {self.scores}")
                if self.config.wandb_on:
                    log_wandb(responses, rewards, self.available_uids)
            else:
                print_info(self, "Market Open")
        else:
            print_info(self, "Market Closed")

    def save_state(self):
        """Saves the state of the validator to a file."""

        state_path = os.path.join(self.config.full_path, "state.pt")
        state = {
            "scores": self.scores,
            "MinerHistory": self.MinerHistory,
            "moving_average_scores": self.moving_average_scores,
        }
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        bt.logging.info(f"Saved {self.config.neuron.name} state.")

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        state_path = os.path.join(self.config.full_path, "state.pt")
        bt.logging.info(f"State path: {state_path}")
        if not os.path.exists(state_path):
            bt.logging.info("Skipping state load due to missing state.pt file.")
            self.scores = [0.0] * len(self.metagraph.S)
            self.moving_average_scores = {uid: 0 for uid in self.metagraph.uids}
            self.MinerHistory = {uid: MinerHistory(uid) for uid in self.available_uids}
            self.timestamp = get_before(minutes=60)
            return
        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)
            self.scores = state["scores"]
            self.MinerHistory = state["MinerHistory"]
            self.moving_average_scores = state["moving_average_scores"]
            all_dates = [mh.latest_timestamp() for mh in self.MinerHistory.values()]
            self.timestamp = max(all_dates)
        except Exception as e:
            bt.logging.error(f"Failed to load state with error: {e}")
