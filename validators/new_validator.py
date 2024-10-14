import os
import bittensor as bt
from validators.helpers import parse_arguments
from validators.oracle import Oracle
import asyncio
import argparse


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
        config = Config(args)
        self.config = config
        bt.logging.info(f"Config: {self.config}")
        self.config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                self.config.logging.logging_dir,
                self.config.wallet.name,
                self.config.wallet.hotkey_str,
                self.config.netuid,
                'validator',
            )
        )
        os.makedirs(self.config.full_path, exist_ok=True)

    def main(self):
        self.config.wallet = bt.wallet(name=self.configconfig.wallet.name, hotkey=self.config.wallet.hotkey)
        self.config.dendrite = bt.dendrite(wallet=self.config.wallet)
        bt.logging.info(f"Config: {vars(self.config)}")
        loop = asyncio.get_event_loop()
        oracle = Oracle(config=self.config)
        try:
            loop.run_forever()
        except KeyboardInterrupt:
            bt.logging.info("Keyboard interrupt detected. Exiting validator.")


# Run the validator.
if __name__ == "__main__":
    validator = Validator()
    validator.main()
