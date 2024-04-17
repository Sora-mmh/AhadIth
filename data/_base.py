from datasets import Dataset
from pathlib import Path
import pandas as pd

from utils._base import create_prompt


class DataLoader:
    def __init__(self, dataset_pth: Path) -> None:
        dataset_df = pd.read_csv(dataset_pth.as_posix())
        dataset_df = dataset_df.rename(
            columns={"text_en": "response", "prompt_for_hadiths": "question"}
        )
        self._dataset = Dataset.from_pandas(dataset_df)
        self._formatted_dataset = self._dataset.map(
            create_prompt,
            remove_columns=["hadith_id", "source", "chapter_no"],
        )
