import argparse
import os

import bittensor as bt
from loguru import logger


def check_config(cls, config: "bt.Config"):
    r"""Checks/validates the config namespace object."""
    bt.logging.check_config(config)

    full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,  # TODO: change from ~/.bittensor/miners to ~/.bittensor/neurons
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            config.neuron.name,
        )
    )
    print("full path:", full_path)
    config.neuron.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.neuron.full_path):
        os.makedirs(config.neuron.full_path, exist_ok=True)

    if not config.neuron.dont_save_events:
        # Add custom event logger for the events.
        logger.level("EVENTS", no=38, icon="📝")
        logger.add(
            os.path.join(config.neuron.full_path, "events.log"),
            rotation=config.neuron.events_retention_size,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            level="EVENTS",
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        )


def add_args(cls, parser):
    """
    Adds relevant arguments to the parser for operation.
    """
    # Netuid Arg: The netuid of the subnet to connect to.
    parser.add_argument("--netuid", type=int, help="Subnet netuid", default=1)

    neuron_type = "validator" if "miner" not in cls.__name__.lower() else "miner"
    bt.logging.debug(f"Determined neuron_type: {neuron_type}")

    # MINER AND VALIDATOR CONFIG
    parser.add_argument(
        "--neuron.name",
        type=str,
        help="Trials for this neuron go in neuron.root / (wallet_cold - wallet_hot) / neuron.name. ",
        default=neuron_type,
    )

    parser.add_argument(
        "--neuron.device",
        type=str,
        help="Device to run on.",
        default="cpu",
    )

    parser.add_argument(
        "--neuron.epoch_length",
        type=int,
        help="The default epoch length (how often we set weights, measured in 12 second blocks).",
        default=25,
    )

    parser.add_argument(
        "--neuron.events_retention_size",
        type=str,
        help="Events retention size.",
        default="2 GB",
    )

    parser.add_argument(
        "--neuron.dont_save_events",
        action="store_true",
        help="If set, we dont save events to a log file.",
        default=False,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=16,
        help="timeout in seconds",
    )

    # VALIDATOR ONLY CONFIG
    if neuron_type == "validator":
        bt.logging.debug("Adding validator-specific arguments")
        parser.add_argument(
            "--neuron.num_concurrent_forwards",
            type=int,
            help="The number of concurrent forwards running at any time.",
            default=1,
        )

        parser.add_argument(
            "--neuron.sample_size",
            type=int,
            help="The number of miners to query in a single step.",
            default=40,
        )

        parser.add_argument(
            "--neuron.disable_set_weights",
            action="store_true",
            help="Disables setting weights.",
            default=False,
        )

        parser.add_argument(
            "--neuron.moving_average_alpha",
            type=float,
            help="Moving average alpha parameter, how much to add of the new observation.",
            default=0.05,
        )

        parser.add_argument(
            "--neuron.axon_off",
            "--axon_off",
            action="store_true",
            # Note: the validator needs to serve an Axon with their IP or they may
            #   be blacklisted by the firewall of serving peers on the network.
            help="Set this flag to not attempt to serve an Axon.",
            default=False,
        )

        parser.add_argument(
            "--neuron.vpermit_tao_limit",
            type=int,
            help="The maximum number of TAO allowed to query a validator with a vpermit.",
            default=4096,
        )

    # MINER ONLY CONFIG
    else:
        bt.logging.debug("Adding miner-specific arguments")
        parser.add_argument(
            "--blacklist.force_validator_permit",
            action="store_true",
            help="If set, we will force incoming requests to have a permit.",
            default=True,
        )

        parser.add_argument(
            "--blacklist.allow_non_registered",
            action="store_true",
            help="If set, miners will accept queries from non registered entities. (Dangerous!)",
            default=False,
        )

        parser.add_argument("--model", type=str, help="The file name of the model that the miner loads weights from")

        parser.add_argument("--hf_repo_id", type=str, help="The Huggingface repo id where the weights file exists")

        parser.add_argument(
            "--validator.min_stake",
            type=int,
            default=8192,  # Multiple of 2**X
            help="Minimum validator stake to accept forward requests from as a miner",
        )


def config(cls):
    """
    Returns the configuration object specific to this miner or validator after adding relevant arguments.
    """
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    cls.add_args(parser)
    return bt.config(parser)