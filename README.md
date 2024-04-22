<div align="center">
<img src="accelerate.png" />
  
# **Foundry S&P 500 Oracle** <!-- omit in toc -->
---
---
---
## Foundry Accelerate <!-- omit in toc -->

[Website](https://foundrydigital.com/accelerate/) • [Validator](https://taostats.io/validators/foundry/) • [Twitter](https://x.com/FoundryServices?s=20)
</div>

---
- [Introduction](#introduction)
- [Design Decisions](#design-decisions)
- [Installation](#installation)
  - [Install PM2](#install-pm2)
  - [Install Repo](#install-repo)
- [About the Rewards Mechanism](#about-the-rewards-mechanism)
- [Road Map](#road-map)
- [License](#license)

---
## Introduction

Foundry is launching Foundry S&P 500 Oracle to incentivize miners to make predictions on the S&P 500 price frequently throughout trading hours. Validators send miners a timestamp (a future time), which the miners need to use to make predictions on the close price of the S&P 500 for the next six 5m intervals. Miners need to respond with their prediction for the price of the S&P 500 at the given time. Validators store the miner predictions, and then calculate the scores of the miners after the predictions mature. Miners are ranked against eachother, naturally incentivizing competition between the miners. 

---
## Design Decisions

A Bittensor integration into financial markets will expose Bittensor to the largest system in the world; the global economy. The S&P 500 serves as a perfect starting place for financial predictions given its utility and name recognition. Financial market predictions were chosen for three main reasons:
1) __Utility:__ financial markets provide a massive userbase of professional traders, wealth managers, and individuals alike
2) __Objective Rewards Mechanism:__ by tying the rewards mechanism to an external source of truth (yahoo finance's S&P Price), the defensibility of the subnet regarding gamification is quite strong. 
3) __Adversarial Environment:__ the adversarial environment, especially given the rewards mechanism, will allow for significant diversity of models. Miners will be driven to acquire different datasets, implement different training methods, and utilize different model architectures in order to develop the most performant models. 
---
## Installation
### Install PM2
First, install PM2:
```
sudo apt update
sudo apt install nodejs npm
sudo npm install pm2@latest -g
```
Verify installation:
```
pm2 --version
```
### Compute Requirements

| Validator |   Miner   |
|---------- |-----------|
|  8gb RAM  |  8gb RAM  | 
|  2 vCPUs  |  2 vCPUs  | 

### Install-Repo

Begin by creating and sourcing a python virtual environment:
```
python3 -m venv .sn28
source .sn28/bin/activate
```
Clone the Foundry S&P 500 Oracle repo:
```
git clone https://github.com/foundryservices/snpOracle.git
```
Install Requirements:
```
pip3 install -e snpOracle
```

### Running a Miner
ecosystem.config.js files have been created to make deployment of miners and validators easier for the node operator. These files are the default configuration files for PM2, and allow the user to define the environment & application in a cleaner way. IMPORTANT: Make sure your have activated your virtual environment before running your validator/miner. 
First copy the .env.template file to .env
```
cp .env.template .env
```
Update the .env file with your Huggingface access token. A huggingface access token can be procured from the huggingface platform. Follow the <a href='https://huggingface.co/docs/hub/en/security-tokens'>steps mentioned here</a> to get your huggingface access token. If you're model weights are uploaded to a repository of your own or if you're reading a custom model weights file from huggingface, make sure to also make changes to the miner.config.js file's --hf_repo_id and --model args.

To run your miner:
```
pm2 start miner.config.js
```
The miner.config.js has few flags added. Any standard flags can be passed, for example, wallet name and hotkey name will default to "default"; if you have a different configuration, edit your "args" in miner.config.js. Below shows a miner.config.js with extra configuration flags.
-  The hf_repo_id flag will be used to define which huggingface model repository the weights file needs to be downloaded from. You need not necessarily load your model weights from huggingface. If you would like to load weights from a local folder like `mining_models/`, then store the weights in the `mining_models/` folder and make sure to define the --hf_repo_id arg to `LOCAL` like `--hf_repo_id LOCAL`.
- The model flag is used to reference a new model you save to the mining_models directory or to your huggingface hf_repo_id. The example below uses the default which is the new base lstm on <a href='https://huggingface.co/foundryservices/bittensor-sn28-base-lstm/tree/main'>Foundry's Huggingface repository</a>.
```
module.exports = {
  apps: [
    {
      name: 'miner',
      script: 'python3',
      args: './neurons/miner.py --netuid 28 --logging.debug --logging.trace --subtensor.network local --wallet.name walletName --wallet.hotkey hotkeyName --axon.port 8091 --hf_repo_id foundryservices/bittensor-sn28-base-lstm --model mining_models/base_lstm_new.h5'
    },
  ],
};
```
### Running a Validator
ecosystem.config.js files have been created to make deployment of miners and validators easier for the node operator. These files are the default configuration files for PM2, and allow the user to define the environment & application in a cleaner way. IMPORTANT: Make sure your have activated your virtual environment before running your validator/miner. 

To run your validator:
```
pm2 start validator.config.js
```

The validator.config.js has few flags added. Any standard flags can be passed, for example, wallet name and hotkey name will default to "default"; if you have a different configuration, edit your "args" in miner.config.js. Below shows a miner.config.js with extra configuration flags. 
```
module.exports = {
  apps: [
    {
      name: 'validator',
      script: 'python3',
      args: './neurons/validator.py --netuid 28 --logging.debug --logging.trace --subtensor.network local --wallet.name walletName --wallet.hotkey hotkeyName'
    },
  ],
};
```

### Running Miner/Validator in Docker
As an alternative to using pm2, a docker image has been pushed to docker hub that can be used in accordance with docker-compose.yml, or the image can be built locally using the Dockerfile in this repo. To prepare the docker compose file, make the following changes to the compose script:
```
version: '3.7'

services:
  my_container:
    image: zrross11/snporacle:1.0.4
    container_name: subnet28-<MINER OR VALIDATOR>
    network_mode: host
    volumes:
      - /home/ubuntu/.bittensor:/root/.bittensor
    restart: always
    command: "python ./neurons/<MINER OR VALIDATOR>.py --wallet.name <YOUR WALLET NAME> --wallet.hotkey <YOUR WALLET HOTKEY> --netuid 28 --axon.port <YOUR AXON PORT> --subtensor.network local --subtensor.chain_endpoint 127.0.0.1:9944 --logging.debug"
```

Once this is ready, run ```docker compose up -d``` in the base directory

## About the Rewards Mechanism

The simplicity of the rewards mechanism is quite intentional. There are no methods to require a machine learning model be run by the miners. This is because the nature of the problem is such that machine learning models will inherently perform better than any method of gamification. By effectively performing a commit-reveal on a future S&P Price Prediction, S&P Oracle ensures that only well-tuned models will survive. 

Root Mean Squared Error(RMSE) is calculated as such:
![image](https://github.com/teast21/snpOracle/assets/109384972/214b9b12-2563-498c-8f06-956c9f9ee7b0)

Directional accuracy is another metric that is calculated using the predictions and the actual close price data. If the direction in which the (n+1)th prediction goes in, from the (n)th prediction is the same as the direction in which the (n+1)th actual price goes in, from the (n)th actual price, then it is considered directionally correct. Directional accuracy is calculated for the predictions (1,2), (2,3), (3,4), (4,5) & (5,6). The directional accuracy is a score between 0 to 100.

The RMSE + directional accuracy is used to compute the rewards and is given a 50-50 weightage. These values together are then normalized to enforce scores between 0 and 1, and those scores are used to update the existing scores in the metagraph. The miners with the worst scores (highest scores) will be rewarded the least. The weighting function applied to how scores are added to the metagraph creates a pseudo-rolling average score for predictions. Thus, a miner will have perfect trust after a perfect prediction, and will also not have 0 trust after having the worst prediction of an epoch. Consistent high-quality performance will result in high trust, and consistent low-quality performance will result in low trust and eventual de-registration. 

---

## Roadmap

Foundry will constantly work to make this subnet more robust, with the north star of creating end-user utility in mind. Some key features we are focused on rolling out to improve the S&P 500 Oracle are listed here:
- [X] Huggingface Integration
- [X] Add Features to Rewards Mechanism
- [X] Query all miners instead of a random subset
- [X] Automate holiday detection and market close logic
- [ ] (Ongoing) Wandb Integration
- [ ] (Ongoing) Altering Synapse to hold short term history of predictions
- [ ] (Ongoing) Front end for end-user access
- [ ] Add new Synapse type for Inference Requests

We happily accept community feedback and features suggestions. Please reach out to @0xthebom on discord :-)

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Foundry Digital LLC

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
