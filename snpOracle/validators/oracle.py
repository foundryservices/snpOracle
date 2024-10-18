import asyncio
import os
import pickle
from datetime import datetime, timedelta

import bittensor as bt
from numpy import full, nan
from pytz import timezone
from substrateinterface import SubstrateInterface

from snpOracle.protocol import Challenge
from snpOracle.utils import check_uid_availability, is_query_time, log_wandb, market_is_open, print_info, setup_bittensor_objects, setup_wandb
from snpOracle.validators.reward import get_rewards


class Oracle:
    neuron_type: str = "Validator"

    def __init__(self, config=None, loop=None):
        self.config = config
        self.loop = loop
        self.prediction_interval = self.config.prediction_interval  # in minutes
        self.N_TIMEPOINTS = self.config.N_TIMEPOINTS  # number of timepoints to predict
        setup_bittensor_objects(self)
        self.axon = bt.axon(wallet=self.wallet)
        self.axon.serve(netuid=self.config.netuid, subtensor=self.subtensor)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network}")
        if self.config.reset_state:
            self.save_state()
            self.scores = [0.0] * len(self.metagraph.S)
            self.moving_avg_scores = self.scores
        else:
            self.load_state()
            self.moving_avg_scores = self.scores
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        self.current_block = self.node_query("System", "Number", [])
        self.blocks_since_last_update = self.current_block - self.node_query("SubtensorModule", "LastUpdate", [self.config.netuid])[self.my_uid]
        self.tempo = self.node_query("SubtensorModule", "Tempo", [self.config.netuid])
        self.set_weights_rate = 100  # in blocks
        self.hotkeys = self.metagraph.hotkeys
        if self.config.wandb_on:
            setup_wandb(self)
        self.available_uids = asyncio.run(self.get_available_uids())
        self.past_predictions = {uid: full((self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan) for uid in self.available_uids}
        self.lock = asyncio.Lock()
        self.loop.create_task(self.scheduled_prediction_request())
        self.loop.create_task(self.refresh_metagraph())
        self.loop.create_task(self.set_weights_loop())

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_state()

    async def get_available_uids(self):
        miner_uids = []
        for uid in range(len(self.metagraph.S)):
            uid_is_available = check_uid_availability(self.metagraph, uid, self.config.vpermit_tao_limit)
            if uid_is_available:
                miner_uids.append(uid)
        return miner_uids

    async def refresh_metagraph(self):
        while True:
            await self.resync_metagraph()
            await asyncio.sleep(120)

    async def set_weights_loop(self):
        while True:
            await self.set_weights()
            await asyncio.sleep(12)

    async def resync_metagraph(self):
        """Resyncs the metagraph and updates the hotkeys and moving averages based on the new metagraph."""
        async with self.lock:
            bt.logging.info("Syncing Metagraph...")
            self.metagraph.sync(subtensor=self.subtensor)
            # Zero out all hotkeys that have been replaced.
            for uid, hotkey in enumerate(self.hotkeys):
                if hotkey != self.metagraph.hotkeys[uid]:
                    self.scores[uid] = 0  # hotkey has been replaced
                    self.past_predictions[uid] = full((self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan)  # reset past predictions
            self.available_uids = asyncio.run(self.get_available_uids())
            # Check to see if the metagraph has changed size.
            # If so, we need to add new hotkeys and moving averages.
            if len(self.hotkeys) < len(self.metagraph.hotkeys) or len(self.scores) < len(self.metagraph.S):
                # Update the size of the moving average scores.
                new_moving_average = [0.0] * len(self.metagraph.S)
                min_len = min(len(self.hotkeys), len(self.scores))
                new_moving_average[:min_len] = self.scores[:min_len]
                self.moving_average = new_moving_average
                self.scores = new_moving_average
                self.hotkeys = self.metagraph.hotkeys
            bt.logging.info("Metagraph updated, re-syncing hotkeys, dendrite pool and moving averages")
            await self.save_state()

    def query_miners(self):
        timestamp = datetime.now(timezone("America/New_York")).isoformat()
        synapse = Challenge(
            timestamp=timestamp,
            prediction_interval=self.prediction_interval,
            N_TIMEPOINTS=self.N_TIMEPOINTS,
        )
        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=[self.metagraph.axons[uid] for uid in self.available_uids],
            synapse=synapse,
            deserialize=False,
        )
        return responses, timestamp

    def node_query(self, module, method, params):
        try:
            result = self.node.query(module, method, params).value
        except Exception:
            # reinitilize node
            self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
            result = self.node.query(module, method, params).value
        return result

    async def set_weights(self):
        # set weights once every tempo + 1
        async with self.lock:
            if self.blocks_since_last_update > self.set_weights_rate:
                total = sum(self.scores)
                if total == 0:
                    total = 1  # prevent division by zero
                weights = [score / total for score in self.scores]
                bt.logging.info(f"Setting weights: {weights}")
                # Update the incentive mechanism on the Bittensor blockchain.
                result, msg = self.subtensor.set_weights(
                    netuid=self.config.netuid,
                    wallet=self.wallet,
                    uids=self.metagraph.uids,
                    weights=weights,
                    wait_for_inclusion=True,
                )
                if result:
                    bt.logging.info("set_weights on chain successfully!")
                else:
                    bt.logging.debug(
                        "Failed to set weights this iteration with message:",
                        msg,
                    )

    async def scheduled_prediction_request(self):
        timestamp = (datetime.now(timezone("America/New_York")) - timedelta(minutes=self.prediction_interval)).isoformat()
        while True:
            try:
                if market_is_open():
                    # how many seconds since 9:30 am EST
                    query_lag = datetime.now(timezone("America/New_York")) - datetime.fromisoformat(timestamp)
                    if is_query_time(self.prediction_interval, timestamp) or query_lag > timedelta(minutes=self.prediction_interval):
                        responses, timestamp = self.query_miners()
                        rewards = get_rewards(
                            self,
                            responses=responses,
                            miner_uids=self.available_uids,
                        )
                        for uid, response, reward in zip(self.available_uids, responses, rewards):
                            if response.prediction is not None:
                                bt.logging.info(f"UID: {uid}  |  Prediction: {response.prediction}  |  Reward: {reward}")
                        # Adjust the scores based on responses from miners and update moving average.
                        for i, value in zip(self.available_uids, rewards):
                            self.moving_avg_scores[i] = (1 - self.config.alpha) * self.moving_avg_scores[i] + self.config.alpha * value

                        self.scores = self.moving_avg_scores.copy()
                        bt.logging.info(f"Scores: {self.scores}")
                        if self.config.wandb_on:
                            log_wandb(responses, rewards, self.available_uids)
                    else:
                        self.current_block = self.node_query("System", "Number", [])
                        self.blocks_since_last_update = (
                            self.current_block - self.node_query("SubtensorModule", "LastUpdate", [self.config.netuid])[self.my_uid]
                        )
                        print_info(self)
                        await asyncio.sleep(12)
                else:
                    bt.logging.info("Market is closed. Sleeping for 2 minutes...")
                    await asyncio.sleep(120)

            except RuntimeError as e:
                bt.logging.error(e)

    async def save_state(self):
        """Saves the state of the validator to a file."""

        state_path = os.path.join(self.config.full_path, "state.pt")
        state = {
            "scores": self.scores,
            "hotkeys": self.hotkeys,
        }
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
        bt.logging.info("Saved validator state.")

    def load_state(self):
        """Loads the state of the validator from a file."""
        bt.logging.info("Loading validator state.")
        state_path = os.path.join(self.config.full_path, "state.pt")
        if not os.path.exists(state_path):
            bt.logging.info("Skipping state load due to missing state.pt file.")
            return
        # backwards compatability with torch version
        # Load the state of the validator from file.
        with open(state_path, "rb") as f:
            state = pickle.load(f)
        self.scores = state["scores"]
        self.hotkeys = state["hotkeys"]
