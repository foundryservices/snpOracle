from typing import Any, List, Union

import bittensor as bt

from snp_oracle.predictionnet.api import SubnetsAPI
from snp_oracle.predictionnet.protocol import Challenge


class PredictionAPI(SubnetsAPI):
    def __init__(self, wallet: "bt.wallet"):
        super().__init__(wallet)
        self.netuid = 28
        self.name = "prediction"

    def prepare_synapse(self, timestamp: str) -> Challenge:
        synapse = Challenge(timestamp=timestamp)
        return synapse

    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> List[int]:
        outputs = []
        for response in responses:
            if response.dendrite.status_code != 200:
                outputs.append(None)
            else:
                outputs.append(response.prediction)
        return outputs
