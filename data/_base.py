from datasets import load_dataset
from typing import Dict


class DataLoader:
    def __init__(self, dataset_name: str = "databricks/databricks-dolly-15k") -> None:
        self._dataset = load_dataset(dataset_name, split="train")
        self._formatted_dataset = self._dataset.map(
            create_prompt,
            remove_columns=["instruction", "context", "respone", "category"],
        )  # to precise


# databricks/databricks-dolly-15k dataset
def create_prompt(sample: Dict):
    bos_token, eos_token = "<s>", "</s>"
    instruction, context, response_row = (
        sample["instruction"],
        sample["context"],
        sample["response"],
    )
    text_row = f"""[INST] Below is the question based on the context. Question: {instruction}. Below is the given the context {context}. Write a response that appropriately completes the request.[/INST]"""
    sample["prompt"] = bos_token + text_row
    sample["completion"] = response_row + eos_token
    return sample

def format_prompt(sample: Dict):
    output_texts = []
    for idx in range(len(sample["prompt"])):
        text = f"{sample["prompt"][idx]}\n\n ### Answer: {sample["completion"][idx]}"
        output_texts.append(text)
        return output_texts
