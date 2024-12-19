module.exports = {
  apps: [
    {
      name: 'validator',
      script: 'python3',
      args: './snp_oracle/neurons/validator.py --netuid 28 --logging.debug --logging.trace --subtensor.network local --wallet.name walletName --wallet.hotkey hotkeyName --neuron.organization huggingfaceOrganization'
    },
  ],
};
