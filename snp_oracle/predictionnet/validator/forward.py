import time
from datetime import datetime, timedelta

import bittensor as bt
import wandb
from numpy import full, nan
from pytz import timezone

import snp_oracle.predictionnet as predictionnet
from snp_oracle.predictionnet.utils.uids import check_uid_availability
from snp_oracle.predictionnet.validator.reward import get_rewards


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
    models_confirmed = self.confirm_models(responses, miner_uids)
    bt.logging.info(f"Models Confirmed: {models_confirmed}")
    rewards = [0 if not model_confirmed else reward for reward, model_confirmed in zip(rewards, models_confirmed)]
    # Check base validator file
    self.update_scores(rewards, miner_uids)