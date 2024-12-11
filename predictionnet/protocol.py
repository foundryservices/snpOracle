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

from typing import List, Optional

import bittensor as bt
import pydantic
from pydantic import SecretStr

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the dummy miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a dummy response.

# ---- miner ----
# Example usage:
#   def dummy( synapse: Dummy ) -> Dummy:
#       synapse.dummy_output = synapse.dummy_input + 1
#       return synapse
#   axon = bt.axon().attach( dummy ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   dummy_output = dendrite.query( Dummy( dummy_input = 1 ) )
#   assert dummy_output == 2


class Challenge(bt.Synapse):
    """
    Protocol for handling encrypted prediction challenges between miners and validators.
    Includes secure handling of decryption keys and manages model/data references.

    Attributes:
        repo_id: Repository identifier where the model is stored
        model: Identifier for the specific model to use
        decryption_key: Securely stored key for decrypting data/models
        data: Identifier for the data to be used
        timestamp: Time at which the validation is taking place
        prediction: List of predicted values for next 6 5m candles
    """

    repo_id: Optional[str] = pydantic.Field(
        default=None,
        title="Repo ID",
        description="Storage repository of the model",
    )

    model: Optional[str] = pydantic.Field(
        default=None,
        title="Model ID",
        description="Which model to use",
    )

    decryption_key: Optional[SecretStr] = pydantic.Field(
        default=None,
        title="Decryption Key",
        description="Secure key for decrypting sensitive data/models",
    )

    data: Optional[str] = pydantic.Field(
        default=None,
        title="Data ID",
        description="Which data to use",
    )

    timestamp: str = pydantic.Field(
        ...,
        title="Timestamp",
        description="The time stamp at which the validation is taking place for",
        allow_mutation=False,
    )

    prediction: Optional[List[float]] = pydantic.Field(
        default=None,
        title="Predictions",
        description="Next 6 5m candles' predictions for closing price of S&P 500",
    )

    def deserialize(self) -> List[float]:
        """
        Deserialize the prediction output from the miner.

        Returns:
            List[float]: The deserialized predictions for the next 6 5m candles.
        """
        return self.prediction

    def get_decryption_key(self) -> Optional[str]:
        """
        Safely retrieve the decryption key when needed.

        Returns:
            Optional[str]: The decryption key value if set, None otherwise.
        """
        if self.decryption_key is not None:
            return self.decryption_key.get_secret_value()
        return None
