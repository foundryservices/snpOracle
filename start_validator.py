import subprocess
import time

import snpOracle
from snpOracle.utils import Config, NestedNamespace, get_version, parse_arguments

webhook_url = ""
current_version = snpOracle.__version__


def update_and_restart(config):
    global current_version
    start_command = ["pm2 start python3 -m snpOracle.validators.validator "]
    for key, value in vars(config).items():
        if isinstance(value, NestedNamespace):
            for key_2, value_2 in vars(value).items():
                start_command.append(f" --{key}.{key_2} {value_2} ")
        else:
            start_command.append(f" --{key} {value} ")

    subprocess.run(start_command)
    if args.autoupdate:
        while True:
            latest_version = get_version()
            print(f"Current version: {current_version}")
            print(f"Latest version: {latest_version}")
            if current_version != latest_version and latest_version is not None:
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
    config = Config(args)
    try:
        update_and_restart(config)
    except Exception as e:
        print(e)
