module.exports = {
  apps: [
    {
      name: 'validator',
      script: 'python3',
      args: './neurons/validator.py --netuid 28 --logging.debug --logging.trace --subtensor.network local --wallet.name walletName --wallet.hotkey hotkeyName'
    },
  ],
};
