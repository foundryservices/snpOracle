import os

import wandb


def setup_wandb(self) -> None:
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is not None:
        wandb.init(
            project=f"sn{self.config.netuid}-validators",
            entity="foundryservices",
            config={
                "hotkey": self.wallet.hotkey.ss58_address,
            },
            name=f"validator-{self.my_uid}-{'0.0.1'}",
            resume="auto",
            dir=self.config.full_path,
            reinit=True,
        )
    else:
        print("WANDB_API_KEY not found in environment variables.")


def log_wandb(responses, rewards, miner_uids):
    wandb_val_log = {
        "miners_info": {
            miner_uid: {
                "miner_response": response.prediction,
                "miner_reward": reward,
            }
            for miner_uid, response, reward in zip(miner_uids, responses, rewards.tolist())
        }
    }
    wandb.log(wandb_val_log)
