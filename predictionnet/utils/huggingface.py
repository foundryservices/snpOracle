import os

from huggingface_hub import HfApi, model_info


class HF_interface:
    def __init__(self):
        self.api = HfApi(token=os.getenv("MINER_HF_ACCESS_TOKEN"))
        self.collection_slug = "foundryservices/oracle-674df1e1ba06279e786a0e37"
        self.collection = self.get_models()

    def get_models(self):
        collection = self.api.get_collection(collection_slug=self.collection_slug)
        return collection

    def check_model_exists(self, repo_id, model_id):
        try:
            # Combine repo_id and model_id
            full_model_id = f"{repo_id}/{model_id}"
            # Fetch model information
            model_info(full_model_id)
            return True
        except Exception as e:
            return False, str(e)

    def add_model_to_collection(self, repo_id, model_id):
        self.api.add_collection_item(
            collection_slug=self.collection_slug,
            item_id=f"{repo_id}/{model_id}",
            item_type="model",
            exists_ok=True,
        )

    def update_collection(self, models):
        id_list = [x.item_id for x in self.collection.items]
        for model in models:
            if model.item_id not in id_list:
                self.add_model_to_collection(
                    collection_slug=self.collection_slug, repo_id=model.repo_id, model_id=model.model_id
                )

    def hotkeys_match(self, synapse, model_id, repo_id) -> bool:
        synapse_hotkey = synapse.TerminalInfo.hotkey
        model_metadata = get_model_metadata(repo_id, model_id)
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
