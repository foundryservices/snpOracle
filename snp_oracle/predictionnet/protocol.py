from typing import List, Optional

import bittensor as bt
import pydantic


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

    decryption_key: Optional[bytes] = pydantic.Field(
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

    def get_decryption_key(self) -> Optional[bytes]:
        """
        Safely retrieve the decryption key when needed.

        Returns:
            Optional[bytes]: The raw Fernet key bytes if set, None otherwise.
        """
        return self.decryption_key
