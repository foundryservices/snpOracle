module.exports = {
  apps: [
    {
      name: 'validator',
      script: 'python3',
      args: './neurons/validator.py --netuid 1 --logging.debug --logging.trace --subtensor.network local --wallet.name validator --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.hotkey default'
    },
  ],
};
