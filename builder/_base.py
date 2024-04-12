from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

import config._base as cfg


class Builder:
    def __init__(self, model_name: str = cfg.model_name) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.bnb_params["use_4bit"],
            bnb_4bit_quant_type=cfg.bnb_params["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.bnb_params["bnb_4bit_compute_dtype"]
            ),
            bnb_4bit_use_double_quant=cfg.bnb_params["use_nested_quant"],
        )
        self._baseline_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map=cfg.device_map
        )
        self._baseline_model.config.use_cache = False
        self._baseline_model.config.pretraining_tp = 1
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._tokenizer.pad_token = self.tokenizer.eos_token
        self._tokenizer.padding_side = "right"
