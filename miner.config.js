module.exports = {
  apps: [
    {
      name: 'miner',
      script: 'python3',
      args: './neurons/miner.py --netuid 28 --logging.debug --logging.trace --subtensor.network local --wallet.name 28a --wallet.hotkey 28a00 --axon.port 13500 --hf_repo_id LOCAL --model mining_models/base_lstm_new.h5 --subtensor.chain_endpoint ws://83.126.40.85:9944'
    },
  ],
};

