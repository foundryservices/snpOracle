#!/bin/bash

# Change to the subtensor directory
cd /subtensor

# Run the localnet script
BUILD_BINARY=0 ./scripts/localnet.sh &

# Keep the container running
tail -f /dev/null
