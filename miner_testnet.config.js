module.exports = {
  apps: [
    {
      name: 'miner',
      script: 'python3',
      args: './neurons/miner.py --netuid 93 --logging.debug --logging.trace --subtensor.network test --wallet.name 28_test --wallet.hotkey 28_test00 --axon.port 13000 --hf_repo_id LOCAL --model base_miner/mining_models/arimax_model.pkl'
    },
  ],
};

