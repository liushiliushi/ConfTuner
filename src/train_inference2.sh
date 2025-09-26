

paras="--add_loss_con True --on_policy False --batch_size_testing 4 --do_sample False --temperature 0 --use_peft --peft_method lora --model_name ../../meta-llama/Ministral-8B-Instruct-2410 --output_dir checkpoints/trivia --dataset hotpot_qa --batch_size_training=4 --val_batch_size=4 --generate=llm --lr=5e-5 --loss_type=brier --num_epochs=2 --merge_peft True --use_wandb --resume True --id trivia_0321"
CUDA_VISIBLE_DEVICES=3,4,5,6,7  accelerate launch --num_processes 5 --mixed_precision bf16 uncertainty_sft.py ${paras}
CUDA_VISIBLE_DEVICES=3 python inference.py ${paras} 
# CUDA_VISIBLE_DEVICES=5 python inference.py ${paras} --add_loss_con True --on_policy False --batch_size_testing 4 --do_sample False --temperature 0 --use_peft --peft_method lora --model_name ../../meta-llama/Llama-3.1-8B-Instruct  --output_dir checkpoints/trivia_0220 --dataset gsm8k_dataset --batch_size_training=4 --val_batch_size=4 --generate=llm --lr=5e-5 --loss_type=sot --num_epochs=2 --merge_peft True --use_wandb --resume True --id trivia_gsm_0220
