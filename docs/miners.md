# Miners

# **DO NOT RUN THE BASE MINER ON MAINNET!**

> **The base miner provided in this repo is _not intended_ to be run on mainnet!**
>
> **If you run the base miner on mainnet, you are not guaranteed to earn anything!**
> It is provided as an example to help you build your own custom models!
>

<div align="center">

| This repository is the official codebase<br>for Bittensor Subnet 28 (SN28) v1.0.0+,<br>which was released on February 20th 2024. | **Testnet UID:**  272 <br> **Mainnet UID:**  28 |
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
First copy the `.env.miner.template` file to `.env`

```shell
cp .env.miner.template .env
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

#### Models
In the Makefile, ensure that the `--model` flag points to the local location of the model you would like validators to evaluate. By default, the Makefile is populated with the base miner model:
`--model mining_models/base_lstm_new.h5`

The `--hf_repo_id` flag will determine which hugging face repository your miner models and data will be uploaded to. This repository must be public in order to ensure validator access. A hugging face repository will be created at the provided path under your hugging face user assuming you provide a valid username and valid `MINER_HF_ACCESS_TOKEN` in your `.env` file.

#### Data
The data your model utilizes will be automatically uploaded to hugging face, in the same repository as your model, defined here: `--hf_repo_id`. The data will be encrypted initially. Once the model is evaluated by validators, the data will be decrypted and published on hugging face.

## Deploying a Miner
We highly recommend that you run your miners on testnet before deploying on mainnet. Be sure to reference our [Registration Fee Schedule](https://github.com/foundryservices/snpOracle/wiki/Registration-Fee-Schedule) to decide when to register your miner.

**IMPORTANT**
> Make sure you have activated your virtual environment before running your miner.

### Base miner

1. Run the command:
   ```shell
   make miner
   ```

### Custom Miner

1. Write custom logic inside the existing `forward` function located in the file `snp_oracle/neurons/miner.py`
2. This function should handle how the miner responds to requests from the validator.
   1. See `base_miner.py` for an example.
3. Edit the command in the Makefile.
   1. Add values for `hf_repo_id` and so forth.
4. Run the Command:
   ```
   make miner
   ```
