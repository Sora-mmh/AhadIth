import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, PeftModel

import config._base as cfg


def generate_response(pipe, prompt):
    sequences = pipe(
        f"<s>[INST] {prompt} [/INST]",
        do_sample=True,
        max_new_tokens=200,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1,
    )
    return sequences[0]["generated_text"]


def create_prompt(sample):
    bos_token, eos_token = "<s>", "</s>"
    question, response = (
        sample["question"],
        sample["response"],
    )
    text_row = f"""[INST] Below is the question based on the context. Question: {question}. Write a response that appropriately answeres the question.[/INST]"""
    sample["prompt"] = bos_token + text_row
    sample["completion"] = response + eos_token
    return sample


def format_prompt(sample):
    output_texts = []
    for idx in range(len(sample["prompt"])):
        text = "{} \n\n ###Answer: {}".format(
            sample["prompt"][idx], sample["completion"][idx]
        )
        output_texts.append(text)
        return output_texts


def get_linear_modules(model) -> None:
    lora_modules = set()
    for name, module in model.named_children():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_modules.add(names[0] if len(names) == 1 else -1)
    if "lm_head" in lora_modules:
        lora_modules.remove("lm_head")
    lora_modules = list(lora_modules)
