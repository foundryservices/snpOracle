import subprocess
import time

import snpOracle
from snpOracle.utils import get_version, parse_arguments

webhook_url = ""
current_version = snpOracle.__version__


def update_and_restart(args):
    global current_version
    wandb = "--wandb_on" if args.wandb_on else ""
    start_command = [
        "pm2",
        "start",
        f" --name {args.neuron.name}",
        f"python3 -m snpOracle.validators.validator --wallet.name {args.wallet.name}"
        f" --wallet.hotkey {args.wallet.hotkey}"
        f" --netuid {args.netuid}"
        f" --subtensor.network {args.subtensor.network}"
        f" --subtensor.chain_endpoint {args.subtensor.chain_endpoint}"
        f" --logging.level {args.logging.level}"
        f" --axon.port {args.axon.port}"
        f"{wandb}",
    ]
    subprocess.run(start_command)
    if args.autoupdate:
        while True:
            latest_version = get_version()
            print(f"Current version: {current_version}")
            print(f"Latest version: {latest_version}")
            if (
                current_version != latest_version
                and latest_version is not None
            ):
                print("Updating to the latest version...")
                subprocess.run(["pm2", "delete", args.neuron.name])
                subprocess.run(["git", "reset", "--hard"])
                subprocess.run(["git", "pull"])
                subprocess.run(["pip", "install", "-e", "."])
                subprocess.run(start_command)
                current_version = latest_version
            print("All up to date!")
            time.sleep(300)


if __name__ == "__main__":
    args = parse_arguments()

    try:
        update_and_restart(args)
    except Exception as e:
        print(e)
