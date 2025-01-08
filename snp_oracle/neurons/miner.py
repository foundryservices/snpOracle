import os
import time
import typing

import bittensor as bt
import tensorflow as tf
from cryptography.fernet import Fernet

# ML imports
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

import snp_oracle.predictionnet as predictionnet
from snp_oracle.base_miner.get_data import prep_data, scale_data
from snp_oracle.base_miner.predict import predict
from snp_oracle.predictionnet.base.miner import BaseMinerNeuron
from snp_oracle.predictionnet.utils.miner_hf import MinerHfInterface

load_dotenv()


class Miner(BaseMinerNeuron):
    """
    Your miner neuron class. You should use this class to define your miner's behavior. In particular, you should replace the forward function with your own logic. You may also want to override the blacklist and priority functions according to your needs.

    This class inherits from the BaseMinerNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a miner such as blacklisting unrecognized hotkeys, prioritizing requests based on stake, and forwarding requests to the forward function. If you need to define custom
    """

    def __init__(self, config=None):
        bt.logging.info("Initializing Miner...")
        bt.logging.info(f"Initial config: {config}")

        super(Miner, self).__init__(config=config)

        bt.logging.info(f"Config after super init: {self.config}")
        bt.logging.info(f"Config model path: {self.config.model if self.config else 'No config'}")
        # TODO(developer): Anything specific to your use case you can do here
        self.model_loc = self.config.model
        if self.config.neuron.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This will force TensorFlow to use CPU only

        if not self.config.model:
            bt.logging.error("--model argument is required")
            exit(1)
        if not self.config.hf_repo_id:
            bt.logging.error("--hf_repo_id argument is required")
            exit(1)

        # Initialize HF interface and upload model
        hf_interface = MinerHfInterface(self.config)
        success, metadata = hf_interface.upload_model(
            hotkey=self.wallet.hotkey.ss58_address, model_path=self.config.model, repo_id=self.config.hf_repo_id
        )

        if success:
            bt.logging.success(
                f"Model {self.config.model} uploaded successfully to {self.config.hf_repo_id}: {metadata}"
            )
        else:
            bt.logging.error(f"Model {self.config.model} upload failed to {self.config.hf_repo_id}: {metadata}")

    async def blacklist(self, synapse: predictionnet.protocol.Challenge) -> typing.Tuple[bool, str]:
        """
        Determines whether an incoming request should be blacklisted and thus ignored. Your implementation should
        define the logic for blacklisting requests based on your needs and desired security parameters.

        Blacklist runs before the synapse data has been deserialized (i.e. before synapse.data is available).
        The synapse is instead contructed via the headers of the request. It is important to blacklist
        requests before they are deserialized to avoid wasting resources on requests that will be ignored.

        Args:
            synapse (predictionnet.protocol.Challenge): A synapse object constructed from the headers of the incoming request.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the synapse's hotkey is blacklisted,
                            and a string providing the reason for the decision.

        This function is a security measure to prevent resource wastage on undesired requests. It should be enhanced
        to include checks against the metagraph for entity registration, validator status, and sufficient stake
        before deserialization of synapse data to minimize processing overhead.

        Example blacklist logic:
        - Reject if the hotkey is not a registered entity within the metagraph.
        - Consider blacklisting entities that are not validators or have insufficient stake.

        In practice it would be wise to blacklist requests from entities that are not validators, or do not have
        enough stake. This can be checked via metagraph.S and metagraph.validator_permit. You can always attain
        the uid of the sender via a metagraph.hotkeys.index( synapse.dendrite.hotkey ) call.

        Otherwise, allow the request to be processed further.
        """

        # TODO(developer): Define how miners should blacklist requests.

        bt.logging.info("Checking miner blacklist")

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = self.metagraph.S[uid].item()

        if not self.config.blacklist.allow_non_registered and synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            # Ignore requests from un-registered entities.
            bt.logging.trace(f"Blacklisting un-registered hotkey {synapse.dendrite.hotkey}")
            return True, "Unrecognized hotkey"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        bt.logging.info(f"Requesting UID: {uid} | Stake at UID: {stake}")

        if stake <= self.config.validator.min_stake:
            # Ignore requests if the stake is below minimum
            bt.logging.info(
                f"Hotkey: {synapse.dendrite.hotkey}: stake below minimum threshold of {self.config.validator.min_stake}"
            )
            return True, "Stake below minimum threshold"

        if self.config.blacklist.force_validator_permit:
            # If the config is set to force validator permit, then we should only allow requests from validators.
            if not self.metagraph.validator_permit[uid]:
                bt.logging.warning(f"Blacklisting a request from non-validator hotkey {synapse.dendrite.hotkey}")
                return True, "Non-validator hotkey"

        bt.logging.trace(f"Not Blacklisting recognized hotkey {synapse.dendrite.hotkey}")
        return False, "Hotkey recognized!"

    async def priority(self, synapse: predictionnet.protocol.Challenge) -> float:
        """
        The priority function determines the order in which requests are handled. More valuable or higher-priority
        requests are processed before others. You should design your own priority mechanism with care.

        This implementation assigns priority to incoming requests based on the calling entity's stake in the metagraph.

        Args:
            synapse (predictionnet.protocol.Challenge): The synapse object that contains metadata about the incoming request.

        Returns:
            float: A priority score derived from the stake of the calling entity.

        Miners may recieve messages from multiple entities at once. This function determines which request should be
        processed first. Higher values indicate that the request should be processed first. Lower values indicate
        that the request should be processed later.

        Example priority logic:
        - A higher stake results in a higher priority value.
        """
        # TODO(developer): Define how miners should prioritize requests.
        caller_uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)  # Get the caller index.
        prirority = float(self.metagraph.S[caller_uid])  # Return the stake as the priority.
        bt.logging.trace(f"Prioritizing {synapse.dendrite.hotkey} with value: ", prirority)
        return prirority

    async def forward(self, synapse: predictionnet.protocol.Challenge) -> predictionnet.protocol.Challenge:
        """
        Processes the incoming 'Challenge' synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (predictionnet.protocol.Challenge): The synapse object containing the 'Challenge_input' data.

        Returns:
            predictionnet.protocol.Challenge: The synapse object with the 'Challenge_output' field set to twice the 'Challenge_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        bt.logging.info(
            f"👈 Received prediction request from: {synapse.dendrite.hotkey} for timestamp: {synapse.timestamp}"
        )

        model_filename = f"{self.wallet.hotkey.ss58_address}/models/model{os.path.splitext(self.config.model)[1]}"

        timestamp = synapse.timestamp
        synapse.repo_id = self.config.hf_repo_id
        synapse.model = model_filename

        if self.config.hf_repo_id == "LOCAL":
            model_path = f"./{self.config.model}"
            bt.logging.info(
                f"Model weights file from a local folder will be loaded - Local weights file path: {self.config.model}"
            )
        else:
            if not os.getenv("MINER_HF_ACCESS_TOKEN"):
                print("Cannot find a Huggingface Access Token - model download halted.")
            token = os.getenv("MINER_HF_ACCESS_TOKEN")
            model_path = hf_hub_download(
                repo_id=self.config.hf_repo_id,
                filename=model_filename,
                use_auth_token=token,
            )
            bt.logging.info(f"Model downloaded from huggingface at {model_path}")

        model = tf.keras.models.load_model(model_path)
        data = prep_data()

        scaler, _, _ = scale_data(data)
        # mse = create_and_save_base_model_lstm(scaler, X, y)

        # type needs to be changed based on the algo you're running
        # any algo specific change logic can be added to predict function in predict.py
        prediction, input_df = predict(timestamp, scaler, model, type="lstm")

        # Generate encryption key for this request
        encryption_key = Fernet.generate_key()

        # Upload encrypted data to HuggingFace
        hf_interface = MinerHfInterface(self.config)
        success, metadata = hf_interface.upload_data(
            hotkey=self.wallet.hotkey.ss58_address,
            data=input_df,
            repo_id=self.config.hf_repo_id,
            encryption_key=encryption_key,
        )

        if success:
            bt.logging.success(f"Encrypted data uploaded successfully to {metadata['data_path']}")
            synapse.data = metadata["data_path"]  # Store the data path in synapse
            synapse.decryption_key = encryption_key  # Provide key to validator
            bt.logging.info(f"synapse.decryption_key: {synapse.decryption_key}")
            bt.logging.info(f"synapse.data: {synapse.data}")
        else:
            bt.logging.error(f"Data upload failed: {metadata['error']}")

        bt.logging.info(f"Prediction: {prediction}")
        # pred_np_array = np.array(prediction).reshape(-1, 1)

        # logic to ensure that only past 20 day context exists in synapse
        synapse.prediction = list(prediction)

        if synapse.prediction is not None:
            bt.logging.success(f"Predicted price 🎯: {synapse.prediction}")
        else:
            bt.logging.info("No price predicted for this request.")

        return synapse

    def save_state(self):
        pass

    def load_state(self):
        pass

    def print_info(self):
        metagraph = self.metagraph
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)

        # Get all values in one go to avoid multiple concurrent requests
        try:
            current_block = self.block  # Single websocket call
            stake = float(metagraph.S[self.uid])
            trust = float(metagraph.T[self.uid])
            incentive = float(metagraph.I[self.uid])
            emission = float(metagraph.E[self.uid])

            log = (
                "Miner | "
                f"Step:{self.step} | "
                f"UID:{self.uid} | "
                f"Block:{current_block} | "
                f"Stake:{stake:.4f} | "
                f"Trust:{trust:.4f} | "
                f"Incentive:{incentive:.4f} | "
                f"Emission:{emission:.4f}"
            )
            bt.logging.info(log)
        except Exception as e:
            bt.logging.error(f"Error getting miner info: {e}")


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with Miner() as miner:
        while True:
            miner.print_info()
            time.sleep(15)
