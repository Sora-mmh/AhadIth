import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, PeftModel

import config._base as cfg


class PEFTConfig(PeftModel):
    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._linear_modules = self.get_linear_modules()
        self._lora_config = LoraConfig(
            lora_alpha=cfg.lora_params["lora_alpha"],
            lora_dropout=cfg.lora_params["lora_dropout"],
            r=cfg.lora_params["lora_r"],
            target_modules=self._linear_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def get_linear_modules(self) -> None:
        self._lora_modules = set()
        for name, module in self._model.named_children():
            if isinstance(module, bnb.nn.Linear4bit):
                names = name.split(".")
                self._lora_modules.add(names[0] if len(names) == 1 else -1)
        if "lm_head" in self._lora_modules:
            self._lora_modules.remove("lm_head")
        self._lora_modules = list(self._lora_modules)
