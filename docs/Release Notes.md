Release Notes
=============

3.0.1
-----
Released on January 12 2025
- Improvements in validator README instructions
- Add flags to opt into additional functionality


3.0.0
-----
Released on January 10th 2025
- Require open sourcing on HuggingFace
- Update to Bittensor v8.5.1 and leverage CR3
- Leverage Poetry for dependency management
- Enhance README instructions


2.2.1
-----
Released on October 28th 2024
- Added custom nonce verification logic to handle delayed requests
    - [miner/verify](https://github.com/foundryservices/snpOracle/pull/36/files#diff-ea84323e3bc641e8ff34d2075637129176a740dc6f684d735be4485ae8199410R194)
- Added `--timeout` flag to specify the allowed nonce timeout in seconds (default: 16)


2.2.0
-----
Released on October 22nd 2024
- Adopt standard python formatting and linter tools
- Leverage pre-commit hooks to enforce clean software development
- Implement GitHub Actions


2.1.1
-----
Released on October 10th 2024
- Upgrade to Bittensor v7.4.0 due to known latency issues


2.1.0
-----
Released on October 9th 2024
- Begin tracking version release notes
- Upgrade to Bittensor v7.3.1
- Hotfix for incentive mechanism
