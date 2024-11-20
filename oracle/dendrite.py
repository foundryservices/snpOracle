import aiohttp
import asyncio
from typing import Union

from bittensor import dendrite
from bittensor.core.stream import StreamingSynapse
from bittensor.core.synapse import Synapse


DENDRITE_ERROR_MAPPING: dict[Type[Exception], tuple] = {
    aiohttp.ClientConnectorError: ("503", "Service unavailable"),
    asyncio.TimeoutError: ("408", "Request timeout"),
    aiohttp.ClientResponseError: (None, "Client response error"),
    aiohttp.ClientPayloadError: ("400", "Payload error"),
    aiohttp.ClientError: ("500", "Client error"),
    aiohttp.ServerTimeoutError: ("504", "Server timeout error"),
    aiohttp.ServerDisconnectedError: ("503", "Service disconnected"),
    aiohttp.ServerConnectionError: ("503", "Service connection error"),
}
DENDRITE_DEFAULT_ERROR = ("422", "Failed to parse response")

class OracleDendrite(dendrite):

    def process_error_message(
            self,
            synapse: Union["Synapse", "StreamingSynapse"],
            request_name: str,
            exception: Exception,
        ) -> Union["Synapse", "StreamingSynapse"]:
            """
            Handles exceptions that occur during network requests, updating the synapse with appropriate status codes and messages.

            This method interprets different types of exceptions and sets the corresponding status code and
            message in the synapse object. It covers common network errors such as connection issues and timeouts.

            Args:
                synapse (bittensor.core.synapse.Synapse): The synapse object associated with the request.
                request_name (str): The name of the request during which the exception occurred.
                exception (Exception): The exception object caught during the request.

            Returns:
                Synapse (bittensor.core.synapse.Synapse): The updated synapse object with the error status code and message.

            Note:
                This method updates the synapse object in-place.
            """

            self.log_exception(exception)

            error_info = DENDRITE_ERROR_MAPPING.get(type(exception), DENDRITE_DEFAULT_ERROR)
            status_code, status_message = error_info

            if status_code:
                synapse.dendrite.status_code = status_code  # type: ignore
            elif isinstance(exception, aiohttp.ClientResponseError):
                synapse.dendrite.status_code = str(exception.code)  # type: ignore

            message = f"{status_message}: {str(exception)}"
            if isinstance(exception, aiohttp.ClientConnectorError):
                message = f"{status_message} at {synapse.axon.ip}:{synapse.axon.port}/{request_name}"  # type: ignore
                raise exception
            elif isinstance(exception, asyncio.TimeoutError):
                message = f"{status_message} after {synapse.timeout} seconds"
                raise exception

            synapse.dendrite.status_message = message  # type: ignore

            return synapse