import asyncio
import bittensor as bt
from numpy import full, nan
from datetime import datetime, timedelta
from substrateinterface import SubstrateInterface
from protocol import Challenge

class Oracle:
    def __init__(self, config=None):
        self.config = config
        self.loop = asyncio.get_event_loop()
        bt.logging(config=self.config, logging_dir=self.config.full_path)
        bt.logging.info(f"Running validator for subnet: {self.config.netuid} on network: {self.config.subtensor.network} with config:")

        # Initialize wallet.
        self.wallet = bt.wallet(config=self.config)
        self.my_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Wallet: {self.wallet}")

        # Initialize subtensor.
        self.subtensor = bt.subtensor(config=self.config)
        bt.logging.info(f"Subtensor: {self.subtensor}")

        # Initialize dendrite.
        self.dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Dendrite: {self.dendrite}")

        # Initialize metagraph.
        self.metagraph = self.subtensor.metagraph(self.config.netuid)
        bt.logging.info(f"Metagraph: {self.metagraph}")

        # Connect the validator to the network.
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
            bt.logging.error(f"\nYour validator: {self.wallet} is not registered to chain connection: {self.subtensor} \nRun 'btcli register' and try again.")
            exit()
        else:
            # Each validator gets a unique identity (UID) in the network.
            self.my_subnet_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            bt.logging.info(f"Running validator on uid: {self.my_subnet_uid}")
        
        self.scores = [1.0] * len(self.metagraph.S)
        # custom params
        self.last_update = 0
        self.current_block = 0
        self.tempo = self.node_query('SubtensorModule', 'Tempo', [self.config.netuid])
        self.moving_avg_scores = [1.0] * len(self.metagraph.S)
        self.node = SubstrateInterface(url=self.config.subtensor.chain_endpoint)
        self.hotkeys = self.metagraph.hotkeys
        self.setup_wandb(self)
        self.prediction_interval = 5 # in minutes
        self.N_TIMEPOINTS = 6 # number of timepoints to predict
        self.past_predictions = {}
        for uid in range(len(self.metagraph.axons)):
            self.past_predictions[uid] = full((self.N_TIMEPOINTS, self.N_TIMEPOINTS), nan)

    async def initialize_uids_and_capacities(self):
        self.available_uid_to_axons = await self.get_available_uids()

    async def get_available_uids(self):
        """Get a dictionary of available UIDs and their axons asynchronously."""
        await self.dendrite.aclose_session()
        tasks = {uid.item(): self.check_uid(self.metagraph.axons[uid.item()], uid.item()) for uid in
                 self.metagraph.uids}
        results = await asyncio.gather(*tasks.values())

        # Create a dictionary of UID to axon info for active UIDs
        available_uids = {uid: axon_info for uid, axon_info in zip(tasks.keys(), results) if axon_info is not None}

        return available_uids
    
    async def refresh_metagraph(self):
        await self.run_sync_in_async(lambda: self.metagraph.sync())

    async def query_miners(self):
        timestamp = datetime.now(timezone('America/New_York')).isoformat()
        synapse = Challenge(timestamp=timestamp)
        responses = self.dendrite.query(
            # Send the query to selected miner axons in the network.
            axons=self.metagraph.axons,
            synapse=synapse,
            deserialize=False,
        )