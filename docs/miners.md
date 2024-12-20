# Mainnet Miners

# **DO NOT RUN THE BASE MINER ON MAINNET!**

> **The base miner provided in this repo is _not intended_ to be run on mainnet!**
>
> **If you run the base miner on mainnet, you are not guaranteed to earn anything!**
> It is provided as an example to help you build your own custom models!
>

<div align="center">

| This repository is the official codebase<br>for Bittensor Subnet 28 (SN28) v1.0.0+,<br>which was released on February 20th 2024. | **Testnet UID:**  93 <br> **Mainnet UID:**  28 |
| - | - |

</div>

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

Update the `.env` file with your miner's values for the following properties.

```text
MINER_HF_ACCESS_TOKEN='REPLACE_WITH_HUGGINGFACE_ACCESS_KEY'
```

#### HuggingFace Access Token
A huggingface access token can be procured from the huggingface platform. Follow the <a href='https://huggingface.co/docs/hub/en/security-tokens'>steps mentioned here</a> to get your huggingface access token.

#### Makefile
Edit the Makefile with you wallet information.

```text
################################################################################
#                               User Parameters                                #
################################################################################
coldkey = default
miner_hotkey = miner
logging_level = info # options = [info, debug, trace]
```

#### Ports
In the Makefile, we have default ports set to `8091` for validator and `8092` for miner. Please change as-needed.

## Deploying a Miner
We highly recommend that you run your miners on testnet before deploying on mainnet.

**IMPORTANT**
> Make sure you have activated your virtual environment before running your miner.

### Base miner

1. Run the command:
    ```shell
    make miner
    ```

### Custom Miner

1. Run the Command:
    ```
    make miner_custom
    ```
