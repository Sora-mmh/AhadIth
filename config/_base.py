model_name = "mistralai/Mistral-7B-Instruct-v0.1"
new_model_name = "finetuned-mistral-qlora-7B-Instruct-v0.1"

lora_params = {"lora_r": 64, "lora_alpha": 16, "lora_dropout": 0.1}

bnb_params = {
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_quant_type": "nf4",
    "use_nested_quant": True,
}

trainer_params = {
    "output_dir": "/home/mmhamdi/workspace/LLMs/TafssirAI/result",
    "num_train_epochs": 1,
    "fp16": False,
    "bf16": False,
    "train_batch_size_per_gpu": 4,
    "eval_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": 1,
    "max_grad_norm": 0.3,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "optim": "paged_adamw_32bit",
    "lr_scheduler_type": "constant",
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "save_steps": 25,
    "logging_steps": 1,
}

sft_params = {
    "max_seq_length": 1024,
    "packing": False,
}

device_map = {"": 0}
