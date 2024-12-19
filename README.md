<div align="center">
<img src="docs/images/accelerate.png" />

# **Bittensor SN28 - S&P 500 Oracle**

<br>

<div align="center">

| This repository is the official codebase<br>for Bittensor Subnet 28 (SN28) v1.0.0+,<br>which was released on February 20th 2024. | **Testnet UID:**  93 <br> **Mainnet UID:**  28 |
| - | - |

</div>

<br>

|     |     |
| :-: | :-: |
| **Status** | <img src="https://img.shields.io/github/v/release/foundryservices/snpOracle?label=Release" height="25"/> <img src="https://img.shields.io/github/actions/workflow/status/foundryservices/snpOracle/ci.yml?label=Build" height="25"/> <br> <a href="https://github.com/pre-commit/pre-commit" target="_blank"> <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&label=Pre-Commit" height="25"/> </a> <a href="https://github.com/psf/black" target="_blank"> <img src="https://img.shields.io/badge/code%20style-black-000000.svg?label=Code%20Style" height="25"/> </a> <br> <img src="https://img.shields.io/github/license/foundryservices/snpOracle?label=License" height="25"/> |
| **Activity** | <img src="https://img.shields.io/github/commit-activity/m/foundryservices/snpOracle?label=Commit%20Activity" height="25"/> <img src="https://img.shields.io/github/commits-since/foundryservices/snpOracle/latest/dev?label=Commits%20Since%20Latest%20Release" height="25"/> <br> <img src="https://img.shields.io/github/release-date/foundryservices/snpOracle?label=Latest%20Release%20Date" height="25"/> <img src="https://img.shields.io/github/last-commit/foundryservices/snpOracle/dev?label=Last%20Commit" height="25"/> <br> <img src="https://img.shields.io/github/contributors/foundryservices/snpOracle?label=Contributors" height="25"/> |
| **Compatibility** | <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ffoundryservices%2FsnpOracle%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.python&logo=python&label=Python&logoColor=yellow" height="25"/> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ffoundryservices%2FsnpOracle%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.bittensor&prefix=v&label=Bittensor" height="25"/> <br> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ffoundryservices%2FsnpOracle%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.tensorflow&prefix=v&logo=tensorflow&label=TensorFlow" height="25"/> <img src="https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Ffoundryservices%2FsnpOracle%2Frefs%2Fheads%2Fdev%2Fpyproject.toml&query=%24.tool.poetry.dependencies.yfinance&prefix=v&label=yfinance" height="25"/> |
| **Social** | <a href="https://foundrydigital.com/accelerate/" target="_blank"> <img src="https://img.shields.io/website?url=https%3A%2F%2Ffoundrydigital.com%2Faccelerate%2F&up_message=Foundry%20Accelerate&label=Website" height="25"/> </a> <br> <a href="https://taostats.io/validators/5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2" target="_blank"> <img src="https://img.shields.io/website?url=https%3A%2F%2Ftaostats.io%2Fvalidators%2Ffoundry%2F&up_message=Foundry%20Accelerate&label=Validator" height="25"/> </a> <br> <a href="https://x.com/FoundryServices?s=20" target="_blank"> <img src="https://img.shields.io/twitter/follow/FoundryServices" height="25"/> </a> |

</div>

## Introduction

Foundry Digital is launching the Foundry S&P 500 Oracle. This subnet incentivizes accurate short term price forecasts of the S&P 500 during market trading hours.

Miners use Neural Network model architectures to perform short term price predictions on the S&P 500.

Validators store price forecasts for the S&P 500 and compare these predictions against the true price of the S&P 500 as the predictions mature.

## Usage

<div align="center">

| Miners | Validators |
| :----: | :--------: |
| [TestNet Docs]() | [TestNet Docs]() |
| [MainNet Docs]() | [MainNet Docs]() |

</div>

## Incentive Mechanism
Please read the [incentive mechanism white paper]() to understand exactly how miners are scored and ranked.

For transparency, there are two key metrics detailed in the white paper that will be calculated to score each miner:
1. **Directional Accuracy** - was the prediction in the same direction of the true price?
2. **Mean Absolute Error** - how far was the prediction from the true price?

## Design Decisions
Integration into financial markets will expose Bittensor to the largest system in the world; the global economy. The S&P 500 serves as a perfect starting place for financial predictions given its utility and name recognition. Financial market predictions were chosen for three main reasons:

#### Utility
Financial markets provide a massive userbase of professional traders, wealth managers, and individuals alike.

#### Objective Rewards Mechanism
By tying the rewards mechanism to an external source of truth, the defensibility of the subnet regarding gamification is quite strong.

#### Adversarial Environment
The adversarial environment, especially given the rewards mechanism, will allow for significant diversity of models. Miners will be driven to acquire different datasets, implement different training methods, and utilize different Neural Network architectures in order to develop the most performant models.
