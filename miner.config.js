module.exports = {
  apps: [
    {
      name: 'miner',
      script: 'python3',
      args: './neurons/miner.py --netuid 1 --logging.debug --logging.trace --subtensor.network local --wallet.name miner --subtensor.chain_endpoint ws://127.0.0.1:9946 --wallet.hotkey default --axon.port 8091'
    },
  ],
};

