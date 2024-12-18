from typing import List, Optional

import bittensor as bt
import pydantic


class Challenge(bt.Synapse):
    """
    A simple dummy protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request sent by the validator.
    - dummy_output: An optional integer value which, when filled, represents the response from the miner.
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

    # Required request input, filled by sending dendrite caller.
    timestamp: str = pydantic.Field(
        ...,
        title="Timestamp",
        description="The time stamp at which the validation is taking place for",
        allow_mutation=False,
    )
    # Optional request output, filled by recieving axon.
    prediction: Optional[List[float]] = pydantic.Field(
        default=None,
        title="Predictions",
        description="Next 6 5m candles' predictions for closing price of S&P 500",
    )

    def deserialize(self) -> int:
        """
        Deserialize the dummy output. This method retrieves the response from
        the miner in the form of dummy_output, deserializes it and returns it
        as the output of the dendrite.query() call.

        Returns:
        - int: The deserialized response, which in this case is the value of dummy_output.

        Example:
        Assuming a Dummy instance has a dummy_output value of 5:
        >>> dummy_instance = Dummy(dummy_input=4)
        >>> dummy_instance.dummy_output = 5
        >>> dummy_instance.deserialize()
        5
        """
        return self.prediction
