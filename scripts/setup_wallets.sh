#!/bin/bash

# Create wallets
btcli wallet new_coldkey --wallet.name owner --no-prompt --use-password MY_PASSWORD
btcli wallet new_coldkey --wallet.name miner --no-prompt --use-password MY_PASSWORD
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default --no-prompt --use-password MY_PASSWORD
btcli wallet new_coldkey --wallet.name validator --no-prompt --use-password MY_PASSWORD
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default --no-prompt --use-password MY_PASSWORD

# Mint tokens
btcli wallet faucet --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946 --no-prompt --use-password MY_PASSWORD
btcli wallet faucet --wallet.name validator --subtensor.chain_endpoint ws://127.0.0.1:9946 --no-prompt --use-password MY_PASSWORD

# Create subnet
btcli subnet create --wallet.name owner --subtensor.chain_endpoint ws://127.0.0.1:9946 --no-prompt --use-password MY_PASSWORD

# Register keys
btcli subnet register --wallet.name miner --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946 --no-prompt --use-password MY_PASSWORD
btcli subnet register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946 --no-prompt --use-password MY_PASSWORD

# Add stake
btcli stake add --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946 --no-prompt --use-password MY_PASSWORD

# Register validator on root subnet and boost
btcli root register --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946 --no-prompt --use-password MY_PASSWORD
btcli root boost --netuid 1 --increase 1 --wallet.name validator --wallet.hotkey default --subtensor.chain_endpoint ws://127.0.0.1:9946 --no-prompt --use-password MY_PASSWORD
