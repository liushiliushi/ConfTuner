

This is the official repository for [*ConfTuner: Training Large Language Models to Express Their Confidence Verbally*](https://arxiv.org/abs/2508.18847), accepted by NeurIPS 2025. 
## How ConfTuner works?

ConfTuner is a novel method to fine-tune large language models (LLMs) using a customized loss function called the Tokenized Brier Score. This approach enables LLMs to more accurately verbalize their confidence in natural language, such as expressing percentages like "80% confident."

ConfTuner consists of two steps:

1. Compute Probability Distribution Over Confidence Tokens: Given a prompt that asks the LLM to output the answer and its confidence for a question, this step extracts the model’s probability distribution over a predefined set of confidence tokens.

2. Fine-Tune Based on Tokenized Brier Score: The probability distribution is used to compute a tokenized Brier score against the ground truth correctness of the generated answer, effectively penalizing miscalibrated confidence. We fine-tune the LLM based on the tokenized Brier score.

![ConfTuner Method](images/method.png)

## Setups

1. Clone the repository:
```bash
git clone git@github.com:liushiliushi/Uncertainty_ft.git
cd Uncertainty_ft
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API credentials for GPT-4o evaluation:
```bash
export OPENAI_DEPLOYMENT_NAME='gpt-4o'
export OPENAI_API_KEY='your-api-key-here'  # Replace with your OpenAI API key
```

Note: The evaluation of answer accuracy is performed using GPT-4o. You need to set up your OpenAI API credentials before running any generation or evaluation scripts.

## Dataset Generation

To generate training datasets for different base models, use the following commands:

### Llama-3.1
For training set:
```bash
python generate_response.py \
    --split train \
    --output_dir ../dataset/hotpot_qa/train_llama_temp=0.jsonl \
    --do_sample False \
    --temperature 0 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset hotpot_qa
```

### Qwen
For training set:
```bash
python generate_response.py \
    --split train \
    --output_dir ../dataset/hotpot_qa/train_Qwen_temp=0.jsonl \
    --do_sample False \
    --temperature 0 \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset hotpot_qa
```

### Mistral
For training set:
```bash
python generate_response.py \
    --split train \
    --output_dir ../dataset/hotpot_qa/train_ministral_temp=0.jsonl \
    --do_sample False \
    --temperature 0 \
    --model_name mistralai/Ministral-8B-Instruct-2410 \
    --dataset hotpot_qa
```

### Datasets for GPT-4o

To generate training datasets for GPT-4o, use the following commands:

```bash
python generate_response.py --split train --output_dir ../dataset/hotpot_qa/train_response_gpt.jsonl --do_sample False --temperature 0 --model_name gpt4o --dataset hotpot_qa
```

To generate testing datasets for GPT-4o, use the following commands:

```bash
python generate_response.py --split test --output_dir ../dataset/hotpot_qa/validation_response_gpt.jsonl --do_sample False --temperature 0 --model_name gpt4o --dataset hotpot_qa

python generate_response.py --split test --output_dir ../dataset/trivia_qa/validation_gpt_temp=0.jsonl --do_sample False --temperature 0 --model_name gpt4o --dataset trivia_qa

python generate_response.py --split test --output_dir ../dataset/grade_school_math/data/validation_gpt_temp=0.jsonl --do_sample False --temperature 0 --model_name gpt4o --dataset gsm8k_dataset

python generate_response.py --split test --output_dir ../dataset/truthful_qa/validation_gpt_temp=0.jsonl --do_sample False --temperature 0 --model_name gpt4o --dataset truthful_qa

python generate_response.py --split test --output_dir ../dataset/StrategyQA/validation_gpt_temp=0.jsonl --do_sample False --temperature 0 --model_name gpt4o --dataset strategy_qa
```


## Training

### Training Command

1. For Llama-3.1:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file llama_recipes/configs/ds_config.json \
    uncertainty_sft.py \
    --add_loss_con True \
    --train_coarse False \
    --batch_size_testing 4 \
    --do_sample False \
    --temperature 0 \
    --use_peft \
    --peft_method lora \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir checkpoints/llama_ft \
    --dataset hotpot_qa \
    --batch_size_training=4 \
    --val_batch_size=4 \
    --generate=llm \
    --lr=1e-5 \
    --loss_type=brier \
    --num_epochs=2 \
    --merge_peft True \
    --use_wandb \
```

2. For Qwen:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 accelerate launch \
    --num_processes 6 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file llama_recipes/configs/ds_config.json \
    uncertainty_sft.py \
    --add_loss_con False \
    --train_coarse True \
    --batch_size_testing 4 \
    --do_sample False \
    --temperature 0 \
    --use_peft \
    --peft_method lora \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --output_dir checkpoints/qwen_ft \
    --dataset hotpot_qa \
    --batch_size_training=4 \
    --val_batch_size=4 \
    --generate=llm \
    --lr=1e-5 \
    --loss_type=brier \
    --num_epochs=3 \
    --merge_peft True \
    --use_wandb
```

3. For Mistral:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file llama_recipes/configs/ds_config.json \
    uncertainty_sft.py \
    --add_loss_con False \
    --train_coarse True \
    --batch_size_testing 4 \
    --do_sample False \
    --temperature 0 \
    --use_peft \
    --peft_method lora \
    --model_name mistralai/Ministral-8B-Instruct-2410 \
    --output_dir checkpoints/ministral_ft \
    --dataset hotpot_qa \
    --batch_size_training=4 \
    --val_batch_size=4 \
    --generate=llm \
    --lr=3e-5 \
    --loss_type=brier \
    --num_epochs=2 \
    --merge_peft True
```

Train on GPT-4o's responses:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    --use_deepspeed \
    --deepspeed_config_file llama_recipes/configs/ds_config.json \
    uncertainty_sft.py \
    --add_loss_con False \
    --train_gpt True \
    --train_coarse False \
    --on_policy False \
    --batch_size_testing 4 \
    --do_sample False \
    --temperature 0 \
    --use_peft \
    --peft_method lora \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --output_dir checkpoints/llama_gpt \
    --dataset hotpot_qa \
    --batch_size_training=4 \
    --val_batch_size=4 \
    --generate=llm \
    --lr=1e-4 \
    --loss_type=sot \
    --num_epochs=2 \
    --merge_peft False \
    --use_wandb
```

### Training Parameters

Key parameters for fine-tuning:
- `--add_loss_con`: Enable consistency loss
- `--train_coarse`: Enable training for confidence levels of 0-9
- `--use_peft`: Enable Parameter-Efficient Fine-Tuning
- `--peft_method`: Choose between 'lora' or 'qlora'
- `--batch_size_training`: Training batch size per gpu
- `--val_batch_size`: Validation batch size per gpu
- `--use_wandb`: Enable Weights & Biases logging
- `--loss_type`: Loss function type (e.g., 'brier')
- `--num_epochs`: Number of training epochs
- `--merge_peft`: Merge PEFT weights after training

## Testing

To evaluate the fine-tuned model:


Validate on confidence levels 0-100:
```bash
python inference.py \
    --model_name /path/to/your/checkpoint \
    --dataset dataset_name \
    --use_wandb \
```

Validate on confidence levels high/medium/low:
```bash
python inference.py \
    --model_name /path/to/your/checkpoint \
    --dataset dataset_name \
    --use_wandb \
    --test_linguistic True
```

You can just run these commands to see the performance on all the datasets:

```bash
cd src
./inference.sh /path/to/your/checkpoint
```

To evaluate the model's cascading performance, run:

```bash
python inference_gpt.py \
    --model_name /path/to/your/checkpoint \
    --dataset hotpot_qa or truthful_qa \
    --use_wandb \
```

## Checkpoints

The checkpoints of ConfTuner:

[ConfTuner based on LLAMA](https://huggingface.co/liushiliushi/ConfTuner-LLaMA)

[ConfTuner based on Qwen](https://huggingface.co/liushiliushi/ConfTuner-Qwen)

[ConfTuner based on Ministral](https://huggingface.co/liushiliushi/ConfTuner-Ministral)

