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

import bittensor as bt
import wandb

# Import Validator Template
from predictionnet.protocol import Challenge
from predictionnet.utils.timestamp import datetime_to_iso8601, get_now
from predictionnet.validator.reward import get_rewards


async def forward(self):
    """
    The forward function is called by the validator every time step.
    It is responsible for querying the network and scoring the responses.
    Args:
        self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.
    """
    # TODO(developer): Define how the validator selects a miner to query, how often, etc.
    # get_random_uids is an example method, but you can replace it with your own.

    responses = query_miners(self)

    rewards = get_rewards(self, responses=responses)

    wandb_val_log = {
        "miners_info": {
            miner_uid: {
                "miner_response": response.prediction,
                "miner_reward": reward,
            }
            for miner_uid, response, reward in zip(self.available_uids, responses, rewards.tolist())
        }
    }

    wandb.log(wandb_val_log)

    # Potentially will need some
    bt.logging.info(f"Scored responses: {rewards}")
    # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
    models_confirmed = self.confirm_models(responses)
    bt.logging.info(f"Models Confirmed: {models_confirmed}")
    rewards = [0 if not model_confirmed else reward for reward, model_confirmed in zip(rewards, models_confirmed)]
    # Check base validator file
    self.update_scores(rewards, self.available_uids)


def query_miners(self):
    bt.logging.info("Querying miners...")
    timestamp = datetime_to_iso8601(get_now())
    synapse = Challenge(timestamp=timestamp)
    responses = self.dendrite.query(
        # Send the query to selected miner axons in the network.
        axons=[self.metagraph.axons[uid] for uid in self.available_uids],
        synapse=synapse,
        deserialize=False,
    )
    return responses
