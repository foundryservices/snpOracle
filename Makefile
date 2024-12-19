################################################################################
#                               User Parameters                                #
################################################################################
coldkey = default
validator_hotkey = validator
miner_hotkey = miner
netuid = $(testnet_netuid)
network = $(testnet)


################################################################################
#                             Network Parameters                               #
################################################################################
finney = wss://entrypoint-finney.opentensor.ai:443
testnet = wss://test.finney.opentensor.ai:443
locanet = ws://127.0.0.1:9944

finney_netuid = 28
testnet_netuid = 93
localnet_netuid = 1
logging_level = info # options= ['info', 'debug', 'trace']


################################################################################
#                                 Commands                                     #
################################################################################
metagraph:
	btcli subnet metagraph --netuid $(netuid) --subtensor.chain_endpoint $(network)

register:
	{ \
		read -p 'Wallet name?: ' wallet_name ;\
		read -p 'Hotkey?: ' hotkey_name ;\
		btcli subnet register --netuid $(netuid) --wallet.name "$$wallet_name" --wallet.hotkey "$$hotkey_name" --subtensor.chain_endpoint $(network) ;\
	}

validator:
	pm2 start python --name validator -- ./snp_oracle/neurons/validator.py \
		--wallet.name $(coldkey) \
		--wallet.hotkey $(validator_hotkey) \
		--subtensor.chain_endpoint $(network) \
		--axon.port 8091 \
		--netuid $(netuid) \
		--logging.level $(logging_level)

miner:
	pm2 start python --name miner  -- ./snp_oracle/neurons/miner.py \
		--wallet.name $(coldkey) \
		--wallet.hotkey $(miner_hotkey) \
		--subtensor.chain_endpoint $(network) \
		--axon.port 8092 \
		--netuid $(netuid) \
		--logging.level $(logging_level) \
		--vpermit_tao_limit 2 \
		--hf_repo_id foundryservices/bittensor-sn28-base-lstm \
		--model mining_models/base_lstm_new.h5

