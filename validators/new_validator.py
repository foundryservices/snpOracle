import os
import bittensor as bt
from validators.helpers import parse_arguments, setup_logging
from validators.oracle import Oracle
import asyncio
from pathlib import Path

class Config:
    def __init__(self, args):
        # Add command-line arguments to the Config object
        for key, value in vars(args).items():
            setattr(self, key, value)
    def get(self, key, default=None):
        return getattr(self, key, default)
    

class Validator:
    def __init__(self):
        args = parse_arguments()
        self.config = Config(args)
        bt.logging.info(f"Config: {self.config}")
        full_path = Path(
        f"~/.bittensor/validators/{self.config.wallet.name}/{self.config.wallet.hotkey}/netuid{self.config.netuid}/validator").expanduser()
        full_path.mkdir(parents=True, exist_ok=True)
        self.config.full_path = str(full_path)

    def main(self):
        setup_logging(self.config)
        self.config.wallet = bt.wallet(name=self.config.wallet.name, hotkey=self.config.wallet.hotkey)
        self.config.dendrite = bt.dendrite(wallet=self.config.wallet)
        bt.logging.info(f"Config: {vars(self.config)}")
        loop = asyncio.get_event_loop()
        oracle = Oracle(config=self.config, loop=loop)
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt detected. Exiting validator.")


# Run the validator.
if __name__ == "__main__":
    validator = Validator()
    validator.main()
