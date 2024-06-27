module.exports = {
  apps: [
    {
      name: 'arimax_miner_test28',
      script: 'python3',
      args: './neurons/miner.py --netuid 93 --logging.debug --logging.trace --subtensor.network test --wallet.name 28_test --wallet.hotkey 28_test00 --axon.port 13000 --hf_repo_id LOCAL --model mining_models/arimax_model.pkl'
    },
  ],
};

