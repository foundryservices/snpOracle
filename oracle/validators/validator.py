import asyncio
from pathlib import Path

import bittensor as bt

from oracle.utils.classes import Config
from oracle.utils.general import parse_arguments
from oracle.validators.weight_setter import weight_setter


class Validator:
    def __init__(self):
        args = parse_arguments()
        self.config = Config(args)
        self.config.neuron.type = "Validator"
        full_path = Path(
            f"{self.config.logging.logging_dir}/{self.config.wallet.name}/{self.config.wallet.hotkey}/netuid{self.config.netuid}/validator"
        ).expanduser()
        full_path.mkdir(parents=True, exist_ok=True)
        self.config.full_path = str(full_path)

    async def main(self):
        loop = asyncio.get_event_loop()
        self.weight_setter = weight_setter(config=self.config, loop=loop)
        try:
            loop.run_forever()
            self.weight_setter.condition.notify_all()
        except BrokenPipeError:
            bt.logging.error("Recieved a Broken Pipe substrate error")
            asyncio.run(self.reset_instance())
        except Exception as e:
            bt.logging.error(f"Unhandled exception: {e}")
        finally:
            bt.logging.info("Exiting Validator")
            self.weight_setter.__exit__()

    async def reset_instance(self):
        self.__init__()
        asyncio.run(self.main())


# Run the validator.
if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.main())
