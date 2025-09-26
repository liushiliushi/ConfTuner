#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 Qwen/Qwen2.5-7B-Instruct"
    exit 1
fi

model_name=$1

python inference.py --use_wandb --test_original_model False --do_sample False --temperature 0 --model_name  ${model_name} --output_dir ${model_name}  --dataset trivia_qa
python inference.py --use_wandb --test_original_model False --do_sample False --temperature 0 --model_name  ${model_name} --output_dir ${model_name}  --dataset hotpot_qa
python inference.py --use_wandb --test_original_model False --do_sample False --temperature 0 --model_name  ${model_name} --output_dir ${model_name}  --dataset strategy_qa
python inference.py --use_wandb --test_original_model False --do_sample False --temperature 0 --model_name  ${model_name} --output_dir ${model_name}  --dataset truthful_qa
python inference.py --use_wandb --test_original_model False --do_sample False --temperature 0 --model_name  ${model_name} --output_dir ${model_name}  --dataset gsm8k_dataset
