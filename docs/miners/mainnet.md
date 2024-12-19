# Mainnet Miners

# **DO NOT RUN THE BASE MINER ON MAINNET!**

> **The base miner provided in this repo is _not intended_ to be run on mainnet!**
>
> **If you run the base miner on mainnet, you are not guaranteed to earn anything!**
> It is provided as an example to help you build your own custom models!
>

## Compute Requirements

|   Miner   |
|-----------|
|  8gb RAM  |
|  2 vCPUs  |

## Installation

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

Clone the repository:
```
git clone https://github.com/foundryservices/snpOracle.git
cd snpOracle
```

Create and source a python virtual environment:
```
python3 -m venv
source .venv/bin/activate
```

Install the requirements with poetry:
```
pip install poetry
poetry install
```

## Configuration

#### Environment Variables
First copy the `.env.template` file to `.env`

```shell
cp .env.template .env
```

Update the `.env` file with your miner's values.

```text
WANDB_API_KEY='REPLACE_WITH_WANDB_API_KEY'
MINER_HF_ACCESS_TOKEN='REPLACE_WITH_HUGGINGFACE_ACCESS_KEY'
HF_ACCESS_TOKEN='REPLACE_WITH_HUGGINGFACE_ACCESS_KEY'
HF_ACCESS_TOKEN = 'REPLACE_WITH_HUGGINGFACE_ACCESS_KEY'
WANDB_API_KEY = 'REPLACE_WITH_WANDB_API_KEY'
GIT_TOKEN='REPLACE_WITH_GIT_TOKEN'
GIT_USERNAME='REPLACE_WITH_GIT_USERNAME'
GIT_NAME="REPLACE_WITH_GIT_NAME"
GIT_EMAIL="REPLACE_WITH_GIT_EMAIL"
```

#### HuggingFace Access Token
A huggingface access token can be procured from the huggingface platform. Follow the <a href='https://huggingface.co/docs/hub/en/security-tokens'>steps mentioned here</a> to get your huggingface access token.

#### Makefile
Edit the Makefile with you wallet and network information.

```text
## Network Parameters ##
finney = wss://entrypoint-finney.opentensor.ai:443
testnet = wss://test.finney.opentensor.ai:443
locanet = ws://127.0.0.1:9944

testnet_netuid = 93
localnet_netuid = 1
logging_level = trace # options= ['info', 'debug', 'trace']

netuid = $(testnet_netuid)
network = $(testnet)

## User Parameters
coldkey = default
validator_hotkey = validator
miner_hotkey = miner
```

## Deploying a Miner

### Base miner
We highly recommend that you run your miners on testnet before deploying on mainnet.

**IMPORTANT**
> Make sure your have activated your virtual environment before running your miner.

1. Run the command:
    ```shell
    make miner
    ```

### Custom Miner
We highly recommend that you run your miners on testnet before deploying on mainnet.

**IMPORTANT**
> Make sure your have activated your virtual environment before running your miner.

1. Run the Command:
    ```
    make miner_custom
    ```
