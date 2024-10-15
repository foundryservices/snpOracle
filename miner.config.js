module.exports = {
  apps: [
    {
      name: 'miner',
      script: 'python3',
      args: './neurons/miner.py --netuid 28 --logging.debug --logging.trace --subtensor.network local --wallet.name walletName --wallet.hotkey hotkeyName --axon.port 8091 --hf_repo_id foundryservices/bittensor-sn28-base-lstm --model mining_models/base_lstm_new.h5'
    },
  ],
};
