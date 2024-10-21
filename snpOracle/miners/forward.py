import os

import bittensor as bt
from huggingface_hub import hf_hub_download

from snpOracle.protocol import Challenge
from snpOracle.utils import load_model, predict, scale_data


async def forward(self, synapse: Challenge) -> Challenge:
    timestamp = synapse.timestamp
    # Download the file
    if self.config.hf_repo_id == "LOCAL":
        model_path = f"./{self.config.model}"
        bt.logging.info(f"Model weights file from a local folder will be loaded - Local weights file path: {self.config.model}")
    else:
        if not os.getenv("HF_ACCESS_TOKEN"):
            print("Cannot find a Huggingface Access Token - model download halted.")
        token = os.getenv("HF_ACCESS_TOKEN")
        model_path = hf_hub_download(repo_id=self.config.hf_repo_id, filename=self.config.model, use_auth_token=token)
        bt.logging.info(f"Model downloaded from huggingface at {model_path}")

    model = load_model(model_path)
    data = self.download_data()
    scaler, _, _ = scale_data(data)

    # type needs to be changed based on the algo you're running
    # any algo specific change logic can be added to predict function in predict.py
    prediction = predict(timestamp, scaler, model, type="lstm")
    bt.logging.info(f"Prediction: {prediction}")
    # pred_np_array = np.array(prediction).reshape(-1, 1)

    # logic to ensure that only past 20 day context exists in synapse
    synapse.prediction = list(prediction[0])

    if synapse.prediction is None:
        bt.logging.success(f"Predicted price 🎯: {synapse.prediction}")
    else:
        bt.logging.info("No price predicted for this request.")
