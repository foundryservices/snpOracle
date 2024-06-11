module.exports = {
  apps: [
    {
      name: 'miner',
      script: 'python3',
      args: './neurons/miner.py --netuid 28 --logging.debug --logging.trace --subtensor.network test --wallet.name 28_test --wallet.hotkey 28_test00 --axon.port 10994 --hf_repo_id LOCAL --model mining_models/base_lstm_new.h5'
    },
  ],
};

