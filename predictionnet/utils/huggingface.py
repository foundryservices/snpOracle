import os
from typing import List

from huggingface_hub import HfApi, model_info

from predictionnet.protocol import Challenge


class HF_interface:
    def __init__(self):
        token = os.getenv("HF_ACCESS_TOKEN")
        if token is None:
            raise ValueError(
                "Huggingface access token not found in environment variables, set it as 'HF_ACCESS_TOKEN'."
            )
        self.api = HfApi(token=token)
        self.collection_slug = "foundryservices/oracle-674df1e1ba06279e786a0e37"
        self.collection = self.get_models()

    def get_models(self):
        collection = self.api.get_collection(collection_slug=self.collection_slug)
        return collection

    def check_model_exists(self, repo_id, model_id) -> bool:
        try:
            # Combine repo_id and model_id
            full_model_id = f"{repo_id}/{model_id}"
            # Fetch model information
            model_info(full_model_id)
            return True
        except Exception as e:
            return False, str(e)

    def add_model_to_collection(self, repo_id, model_id) -> None:
        self.api.add_collection_item(
            collection_slug=self.collection_slug,
            item_id=f"{repo_id}/{model_id}",
            item_type="model",
            exists_ok=True,
        )

    def update_collection(self, responses: List[Challenge]) -> None:
        id_list = [x.item_id for x in self.collection.items]
        for response in responses:
            if f"{response.repo_id}/{response.model_id}" not in id_list:
                self.add_model_to_collection(
                    collection_slug=self.collection_slug, repo_id=response.repo_id, model_id=response.model_id
                )
        self.collection = self.get_models()

    def hotkeys_match(self, synapse) -> bool:
        synapse_hotkey = synapse.TerminalInfo.hotkey
        model_metadata = get_model_metadata(synapse.repo_id, synapse.model_id)
        if model_metadata:
            model_hotkey = model_metadata.get("hotkey")
            if synapse_hotkey == model_hotkey:
                return True
            else:
                return False
        else:
            return False


def get_model_metadata(repo_id, model_id):
    try:
        model = model_info(f"{repo_id}/{model_id}")
        metadata = model.cardData
        if metadata is None:
            hotkey = None
        else:
            hotkey = metadata.get("hotkey")
        return {"hotkey": hotkey, "timestamp": model.created_at}
    except Exception as e:
        return False, str(e)
