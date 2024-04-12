from datasets import load_dataset
from typing import Dict


from utils._base import create_prompt


class DataLoader:
    def __init__(self, dataset_name: str = "databricks/databricks-dolly-15k") -> None:
        self._dataset = load_dataset(dataset_name, split="train")
        self._formatted_dataset = self._dataset.map(
            create_prompt,
            remove_columns=["instruction", "context", "response", "category"],
        )  # to precise
