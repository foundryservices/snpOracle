from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import bittensor as bt


class SubnetsAPI(ABC):
    def __init__(self, wallet: "bt.wallet"):
        self.wallet = wallet
        self.dendrite = bt.dendrite(wallet=wallet)

    async def __call__(self, *args, **kwargs):
        return await self.query_api(*args, **kwargs)

    @abstractmethod
    def prepare_synapse(self, *args, **kwargs) -> Any:
        """
        Prepare the synapse-specific payload.
        """
        ...

    @abstractmethod
    def process_responses(self, responses: List[Union["bt.Synapse", Any]]) -> Any:
        """
        Process the responses from the network.
        """
        ...

    async def query_api(
        self,
        axons: Union[bt.axon, List[bt.axon]],
        deserialize: Optional[bool] = False,
        timeout: Optional[int] = 12,
        n: Optional[float] = 0.1,
        uid: Optional[int] = None,
        **kwargs: Optional[Any],
    ) -> Any:
        """
        Queries the API nodes of a subnet using the given synapse and bespoke query function.

        Args:
            axons (Union[bt.axon, List[bt.axon]]): The list of axon(s) to query.
            deserialize (bool, optional): Whether to deserialize the responses. Defaults to False.
            timeout (int, optional): The timeout in seconds for the query. Defaults to 12.
            n (float, optional): The fraction of top nodes to consider based on stake. Defaults to 0.1.
            uid (int, optional): The specific UID of the API node to query. Defaults to None.
            **kwargs: Keyword arguments for the prepare_synapse_fn.

        Returns:
            Any: The result of the process_responses_fn.
        """
        synapse = self.prepare_synapse(**kwargs)
        bt.logging.debug(f"Quering valdidator axons with synapse {synapse.name}...")
        responses = await self.dendrite(
            axons=axons,
            synapse=synapse,
            deserialize=deserialize,
            timeout=timeout,
        )
        return self.process_responses(responses)
