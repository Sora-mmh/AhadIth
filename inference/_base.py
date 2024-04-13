import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import config._base as cfg
from utils._base import create_prompt

logging.basicConfig(level=logging.INFO)


class Inference:
    def __init__(self, model_name) -> None:
        self._model_name = model_name
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map=cfg.device_map,
        )
        self._model.config.use_cache = False
        self._model.config.pretraining_tp = 1
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

    def test(self, test_sample) -> None:
        benchmark_test = create_prompt(test_sample)
        test_prompt = benchmark_test["prompt"]
        test_completion = benchmark_test["completion"]
        model_input = self._tokenizer(test_prompt, return_tensors="pt").to("cuda")
        self._model.eval()
        with torch.no_grad():
            completed_answer = self._tokenizer.decode(
                self._model.generate(**model_input, max_new_tokens=256, pad_token_id=2)[
                    0
                ],
                skip_special_tokens=False,
            )
        logging.info(f"The prompt followed by its completion : \n {completed_answer}")
