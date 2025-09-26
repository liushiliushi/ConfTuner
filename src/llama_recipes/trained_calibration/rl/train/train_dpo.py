import argparse
import pdb 
import json
import re 
from tqdm import tqdm 
import time 
from pathlib import Path

import torch
import numpy as np
import random
import wandb

from trained_calibration.rl.dataset.formatting import format_dataset, format_baseline_dataset
from trained_calibration.rl.reward_model import RewardModel
from trained_calibration.rl.train.my_dpo_trainer import MyDPOTrainer

from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, AutoPeftModelForCausalLM


import os 



def main(args):

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.mapping_fxn_key is None:
        mapping_fxn = None
    elif args.mapping_fxn_key == "false_accept_over_false_reject":
        mapping_fxn = {(True,True): 1.0, # correctly accepting is worth full reward
                            (False, False): 1.0, # correct rejection is also worth full reward
                            (True, False): -1.0, # false rejection is WORSE than false acceptance
                            (False, True): -0.5}
    elif args.mapping_fxn_key == "true_rejection_better_than_true_accept":
        mapping_fxn = {(True,True): 1.0, # correctly accepting is worth full reward
                            (False, False): 1.0, # correct rejection is also worth full reward
                            (True, False): -1.0, # false rejection is WORSE than false acceptance
                            (False, True): -0.5} # false acceptance is better
    else:
        mapping_fxn = None

    if not args.do_baseline:
        train_dataset, threshold = format_dataset(args.train_dataset, 
                                                  limit_per_q=args.limit_per_q, 
                                                  balance_types=args.balance_types, 
                                                balance_fp_tp=args.balance_fp_tp,
                                                  mapping_fxn=mapping_fxn)

        # don't balance validation 
        valid_dataset, __ = format_dataset(args.valid_dataset, 
                                           limit_per_q=args.limit_per_q, 
                                           mapping_fxn=mapping_fxn)
    else:
        train_dataset, threshold = format_baseline_dataset(args.train_dataset, limit_per_q=args.limit_per_q)
        valid_dataset, __ = format_baseline_dataset(args.valid_dataset, limit_per_q=args.limit_per_q)

    print(f"Training on {len(train_dataset)} examples")
    print(f"Threshold: {threshold}")
    print(f"Validating on {len(valid_dataset)} examples")
  
    # shuffle first so it's not all one topic 
    if args.shuffle: 
        train_dataset = train_dataset.shuffle()
        valid_dataset = valid_dataset.shuffle()

    if args.train_limit is not None:
        train_dataset = train_dataset.select(range(args.train_limit))

    if args.valid_limit is not None:
        limit = min(args.valid_limit, len(valid_dataset))
        valid_dataset = valid_dataset.select(range(limit))

    wandb.init(
        project="trained_calibration",
        # track hyperparameters and run metadata
        config=args.__dict__)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto"
        # load_in_4bit=True,
        # cache_dir="/nas-ssd2/esteng/.cache",
        # use_safetensors=False
    )


    tokenizer = AutoTokenizer.from_pretrained(args.model) 
    if "llama" in args.model:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # load reward model 
    # NOTE (elias): we do this even with the baseline because we need it to get F1, which we use for model selection
    # so to have a fair comparison, we also select baseline model based on F1
    reward_model = RewardModel(args.reward_model, args.metric_model_device, quantization_config=bnb_config)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        output_dir=args.output_dir,
        report_to="wandb",
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        optim=args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_calibration",
        gradient_checkpointing_kwargs=dict(use_reentrant=args.gradient_checkpointing_use_reentrant),
        seed=args.seed,
        save_total_limit=2,
        metric_for_best_model="eval_f1",
        load_best_model_at_end=True,
    )

    # TODO (elias): for now not using reference model
    dpo_trainer = MyDPOTrainer(
        model,                 # base model from SFT pipeline
        None,             # typically a copy of the SFT trained base model
        beta=0.1,              # temperature hyperparameter of DPO
        train_dataset=train_dataset, # dataset prepared above
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,   # tokenizer
        peft_config=peft_config,
        args=training_args,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        eval_model=reward_model,
        eval_thresh=threshold,
        n_eval_batches=args.n_eval_batches,
        generate_during_eval=True,
    )

    dpo_trainer.train()
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--reward_model", type=str, default="gpt2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_dataset", type=str, default="outputs/tqa_overfit.jsonl")
    parser.add_argument("--valid_dataset", type=str, default="outputs/tqa_overfit.jsonl")
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--valid_limit", type=int, default=60)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument('--validate', action=argparse.BooleanOptionalAction)
    parser.add_argument('--save', action=argparse.BooleanOptionalAction)
    parser.add_argument("--metric_model_device", type=str, default="cuda:0")
    parser.add_argument("--n_eval_batches", type=int, default=10) 
    parser.add_argument("--limit_per_q", type=int, default=3)  

    parser.add_argument("--do_baseline", type=bool, default=False, help="set to true to rank examples only by whether they are correct or not, not by the probability of accept")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--per_device_train_batch_size", type=int, default=3)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--optimizer_type", type=str, default="paged_adamw_32bit")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--balance_types", type=bool, default=False)
    parser.add_argument("--balance_fp_tp", type=bool, default=False)

    parser.add_argument("--mapping_fxn_key", type=str, default=None)

    args = parser.parse_args()
    main(args)
