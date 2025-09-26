# uncertainty_sft.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
import os
import dataclasses
import fire
import random
import torch.optim as optim
from peft import get_peft_model, PeftModel
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
)
from accelerate import Accelerator
from accelerate.utils import is_xpu_available
from warnings import warn

from llama_recipes.configs import train_config as TRAIN_CONFIG
from llama_recipes.configs import quantization_config as QUANTIZATION_CONFIG
from llama_recipes.data.concatenator import ConcatDataset2
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset2, get_dataset_reflection
from llama_recipes.utils.train_utils_uncertainty import (
    train_chat,
    test_vllm,
    test_gpt,
    test_cross,
    test_reflection,
    test_reflection_gpt,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from warnings import warn


def setup_wandb(train_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    if "Ministral" in train_config.model_name:
        from llama_recipes.configs import wandb_config_mini as WANDB_CONFIG_MINI
        wandb_config = WANDB_CONFIG_MINI()
    elif "Qwen" in train_config.model_name:
        from llama_recipes.configs import wandb_config_qwen as WANDB_CONFIG_QWEN
        wandb_config = WANDB_CONFIG_QWEN()
    else:
        from llama_recipes.configs import wandb_config_llama as WANDB_CONFIG_Llama
        wandb_config = WANDB_CONFIG_Llama()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(dataclasses.asdict(train_config), allow_val_change=True)
    return run



def main(**kwargs):

    train_config = TRAIN_CONFIG()
    update_config(train_config, **kwargs)
    
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    else:
        torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    accelerator = Accelerator()

    wandb_run = None
    if train_config.use_wandb and accelerator.is_main_process:
        wandb_run = setup_wandb(train_config, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name, 
        padding_side='left'
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset_test = get_preprocessed_dataset2(tokenizer, 'test', train_config)

    accelerator.print("==============original test================")
    responses = test_gpt_reflection(train_config, dataset_test, tokenizer, wandb_run, original=True)


if __name__ == "__main__":
    fire.Fire(main)