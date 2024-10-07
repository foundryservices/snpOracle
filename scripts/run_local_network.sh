#!/bin/bash

# Start the local Subtensor network
BUILD_BINARY=0 ./subtensor/scripts/localnet.sh &

# Wait for the network to start
sleep 30

# Run setup script
/setup_wallets.sh

# Start the subnet miner
python3 /bittensor-subnet-template/neurons/miner.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name miner --wallet.hotkey default --logging.debug &

# Start the subnet validator
python3 /bittensor-subnet-template/neurons/validator.py --netuid 1 --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.name validator --wallet.hotkey default --logging.debug &

# Keep the container running
tail -f /dev/null
