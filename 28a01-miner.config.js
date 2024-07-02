module.exports = {
  apps: [
    {
      name: 'miner_arimax_28a01',
      script: 'python3',
      args: './neurons/miner.py --netuid 28 --logging.debug --logging.trace --subtensor.network local --wallet.name 28a --wallet.hotkey 28a01 --axon.port 13501 --hf_repo_id LOCAL --model mining_models/arimax_model.pkl --scheduler False --subtensor.chain_endpoint ws://83.126.40.85:9944'
    },
  ],
};

