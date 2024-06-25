module.exports = {
  apps: [
    {
      name: 'miner',
      script: 'python3',
      args: './neurons/miner.py --netuid 28 --logging.debug --logging.trace --subtensor.network local --wallet.name 28a --wallet.hotkey 28a00 --axon.port 13500 --hf_repo_id LOCAL --model base_miner/mining_models/arimax_model.pkl --subtensor.chain_endpoint ws://83.126.40.85:9944'
    },
  ],
};

