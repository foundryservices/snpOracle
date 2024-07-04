module.exports = {
  apps: [
    {
      name: 'arimax_validator_test28',
      script: 'python3',
      args: './neurons/validator.py --netuid 93 --logging.debug --logging.trace --subtensor.network test --wallet.name 28_test --wallet.hotkey 28_test01 --axon.port 13001 --hf_repo_id LOCAL --model mining_models/arimax_model.pkl'
    },
  ],
};
