################################################################################
#                               User Parameters                                #
################################################################################
# coldkey = validator
coldkey = miner
validator_hotkey = default
miner_hotkey = default
netuid = $(localnet_netuid)
network = $(localnet)
logging_level = debug # options= ['info', 'debug', 'trace']

################################################################################
#                             Network Parameters                               #
################################################################################
finney = wss://entrypoint-finney.opentensor.ai:443
testnet = wss://test.finney.opentensor.ai:443
localnet = ws://127.0.0.1:9945

finney_netuid = 28
testnet_netuid = 272
localnet_netuid = 1


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
		--logging.$(logging_level)


miner:
	pm2 start python --name miner  -- ./snp_oracle/neurons/miner.py \
		--wallet.name $(coldkey) \
		--wallet.hotkey $(miner_hotkey) \
		--subtensor.chain_endpoint $(network) \
		--axon.port 8092 \
		--netuid $(netuid) \
		--logging.$(logging_level) \
		--vpermit_tao_limit 2 \
		--blacklist.force_validator_permit true \
		--hf_repo_id your_repo_id \
		--model mining_models/base_lstm_new.h5
