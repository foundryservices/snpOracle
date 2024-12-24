# Validators

<div align="center">

| This repository is the official codebase<br>for Bittensor Subnet 28 (SN28) v1.0.0+,<br>which was released on February 20th 2024. | **Testnet UID:**  93 <br> **Mainnet UID:**  28 |
| - | - |

</div>

## Compute Requirements

| Validator |
| :-------: |
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

Update the `.env` file with your validator's values.

```text
WANDB_API_KEY='REPLACE_WITH_WANDB_API_KEY'
HF_ACCESS_TOKEN='REPLACE_WITH_HUGGINGFACE_ACCESS_KEY'
GIT_TOKEN='REPLACE_WITH_GIT_TOKEN'
GIT_USERNAME='REPLACE_WITH_GIT_USERNAME'
GIT_NAME="REPLACE_WITH_GIT_NAME"
GIT_EMAIL="REPLACE_WITH_GIT_EMAIL"
```

See WandB API Key and HuggingFace setup below.

#### Obtain & Setup WandB API Key
Before starting the process, validators would be required to procure a WANDB API Key. Please follow the instructions mentioned below:<br>

- Log in to <a href="https://wandb.ai">Weights & Biases</a> and generate an API key in your account settings.
- Set the variable `WANDB_API_KEY` in the `.env` file.
- Finally, run `wandb login` and paste your API key. Now you're all set with weights & biases.

#### HuggingFace Access Token
A huggingface access token can be procured from the huggingface platform. Follow the <a href='https://huggingface.co/docs/hub/en/security-tokens'>steps mentioned here</a> to get your huggingface access token.

#### Git Access Token
A git token can be procured from the huggingface platform. Follow the <a href='https://huggingface.co/docs/hub/en/security-tokens'>steps mentioned here</a> to get your huggingface access token. Be sure to scope this token to the organization repository. The `username`, `name`, and `email` environment variable properties are all tied to your HuggingFace account.

## Deploying a Validator
**IMPORTANT**
> Make sure your have activated your virtual environment before running your validator.
1. Run the command:
    ```shell
    make validator
    ```

Inspect the Makefile for port configurations.