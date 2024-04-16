from random import randrange
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, AutoModelForCausalLM
from peft import LoraConfig

import config._base as cfg
from data._base import DataLoader
from builder._base import Builder
from utils._base import format_prompt, get_linear_modules
from merge._base import Merger
from inference._base import Inference

if __name__ == "__main__":
    response_template = "### Answer:"
    builder = Builder()
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=builder._tokenizer
    )
    dataset = DataLoader()
    formatted_dataset = dataset._formatted_dataset
    linear_modules = get_linear_modules(builder._baseline_model)
    if linear_modules is None:
        linear_modules = [
            "q_proj",
            "k_proj",
            "down_proj",
            "gate_proj",
            "o_proj",
            "v_proj",
            "up_proj",
        ]
    peft_config = LoraConfig(
        lora_alpha=cfg.lora_params["lora_alpha"],
        lora_dropout=cfg.lora_params["lora_dropout"],
        r=cfg.lora_params["lora_r"],
        target_modules=linear_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_arguments = TrainingArguments(
        output_dir=cfg.trainer_params["output_dir"],
        num_train_epochs=cfg.trainer_params["num_train_epochs"],
        per_device_train_batch_size=cfg.trainer_params["train_batch_size_per_gpu"],
        per_device_eval_batch_size=cfg.trainer_params["eval_batch_size_per_gpu"],
        gradient_accumulation_steps=cfg.trainer_params["gradient_accumulation_steps"],
        optim=cfg.trainer_params["optim"],
        save_steps=cfg.trainer_params["save_steps"],
        logging_steps=cfg.trainer_params["logging_steps"],
        learning_rate=cfg.trainer_params["learning_rate"],
        weight_decay=cfg.trainer_params["weight_decay"],
        gradient_checkpointing=cfg.trainer_params["gradient_checkpointing"],
        fp16=cfg.trainer_params["fp16"],
        bf16=cfg.trainer_params["bf16"],
        max_grad_norm=cfg.trainer_params["max_grad_norm"],
        max_steps=cfg.trainer_params["max_steps"],
        warmup_ratio=cfg.trainer_params["warmup_ratio"],
        group_by_length=cfg.trainer_params["group_by_length"],
        lr_scheduler_type=cfg.trainer_params["lr_scheduler_type"],
    )
    trainer = SFTTrainer(
        model=builder._baseline_model,
        train_dataset=formatted_dataset,
        formatting_func=format_prompt,
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=cfg.sft_params["max_seq_length"],
        args=training_arguments,
        packing=cfg.sft_params["packing"],
    )
    trainer.train()
    trainer.model.save_pretrained(cfg.new_model_name)
    baseline_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=cfg.device_map,
    )
    merger = Merger(baseline_model, cfg.new_model_name, builder._tokenizer)
    merger.merge()
    merged_model_name = merger._merged_model_name
    del merger
    torch.cuda.empty_cache()
    inference = Inference(merged_model_name)
    inference.test(dataset._dataset[randrange(len(dataset._dataset))])