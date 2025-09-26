# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
# from vllm import LLM
# from vllm.lora.request import LoRARequest
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
import sys
from tqdm import tqdm
from transformers import LlamaTokenizer
import json
import torch.nn.functional as F
import wandb

from llama_recipes.model_checkpointing import save_peft_checkpoint, save_model_checkpoint, save_merged_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
from llama_recipes.utils.compute_metrics import compute_conf_metrics, plot_confidence_histogram, plot_ece_diagram
from llama_recipes.utils.postprocess import postprocess_extract, confidence_replace, confidence_replace_3level, confidence_replace_gpt,confidence_replace_correct
from vllm import LLM, SamplingParams
def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, accelerator=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter simultaneously.")

    if use_profiler:
        wait_step, warmup_step, active_step = 1, 2, 3
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=wait_step, warmup=warmup_step, active=active_step, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profiler_dir),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        warmup_step = cfg.flop_counter_start
        # If you want to distinguish rank in multi-card, you can use accelerator.process_index
        with FlopMeasure(rank=0, warmup_step=warmup_step) as flop_counter:
            yield flop_counter
    else:
        yield None

def train_chat(
    model,
    train_dataloader,
    eval_dataloader,
    test_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    accelerator,
    wandb_run=None,
):   
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    if train_config.use_fp16 and torch.cuda.is_available():
        autocast = torch.cuda.amp.autocast
    else:
        autocast = contextlib.nullcontext 

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    print("----------------------------")
    print(token_ids)
    num_indices = torch.tensor(token_ids).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 80,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.1,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(train_dataloader, colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True, disable=not accelerator.is_local_main_process)
            for step, batch in enumerate(pbar): 
                total_train_steps += 1
                # stop when the maximum number of training steps is reached
                if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                    max_steps_reached = True
                    accelerator.print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)

                y = batch.data.pop('y')

                with autocast():
                    # get the
                    output = model(**batch)
                    logits = output.logits
                    loss_con = output.loss

                num_token = logits[:,-1,:] # get the logit of the confidence token
                del logits
                scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(y.shape[0], 101).to(model.device)

                if train_config.loss_type == 'brier':
                    num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                    num_conf = F.softmax(num_conf, dim=1)
                    # compute the loss
                    y_expanded = y.expand(y.shape[0], 101)
                    squared_differences = (y_expanded - scores) ** 2
                    loss_cal = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
                elif train_config.loss_type == 'sot':
                    norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                    smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                    smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                    loss_cal = -torch.sum(norm_logit * smoothed, dim=1).mean()
                if train_config.add_loss_con:
                    loss = loss_con + loss_cal
                else:
                    loss = loss_cal
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.detach().float()

                if wandb_run and accelerator.is_main_process:
                    wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                            'train/loss_con': loss_con.detach().float(),
                                'train/loss_cal': loss_cal.detach().float()
                        })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    torch.cuda.empty_cache()
 

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        avg_loss = total_loss / len(train_dataloader)
        avg_ppl = float(torch.exp(torch.tensor(avg_loss)))  # ppl = exp(loss)

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.run_validation and accelerator.is_main_process:
            eval_ppl, eval_epoch_loss, _, _ = evaluation_chat(
                model,
                train_config,
                eval_dataloader,
                tokenizer,
                accelerator,
                wandb_run,
            )
            # if eval_epoch_loss < best_val_loss:
            if True:
                best_val_loss = eval_epoch_loss
                if train_config.save_model and accelerator.is_main_process:
                    if train_config.use_peft:
                        if hasattr(model, "module"):
                            model = model.module
                        if train_config.merge_peft and epoch == (train_config.num_epochs - 1):
                            model = model.to(dtype=torch.float32)
                            model = model.merge_and_unload()
                            model = model.to(dtype=torch.float16)
                            save_merged_checkpoint(model, tokenizer, train_config.output_dir)
                            accelerator.print(f"Merged modules are saved in {train_config.output_dir} directory")
                        else:
                            save_peft_checkpoint(model, train_config.output_dir)
                            accelerator.print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        save_model_checkpoint(model, train_config.output_dir)
                        accelerator.print(f"Model is saved in {train_config.output_dir} directory")

    results = None
    sys.exit(0)

    return results

def train_gpt(
    model,
    train_dataloader,
    eval_dataloaders_dict,  # Changed to dictionary of dataset_name -> eval_dataloader
    test_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    accelerator,
    wandb_run=None,
):   
    """
    Trains the model on the given dataloader and evaluates on 5 datasets

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        eval_dataloaders_dict: Dictionary mapping dataset names to their respective eval dataloaders
        test_dataloader: The test dataloader
        tokenizer: The tokenizer used for processing text
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        train_config: The training configuration
        accelerator: The accelerator for distributed training
        wandb_run: The wandb run object for logging

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    if train_config.use_fp16 and torch.cuda.is_available():
        autocast = torch.cuda.amp.autocast
    else:
        autocast = contextlib.nullcontext 

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_eval_score = float(0)
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    print("----------------------------")
    print(token_ids)
    num_indices = torch.tensor(token_ids).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 80,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(train_dataloader, colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True, disable=not accelerator.is_local_main_process)
            for step, batch in enumerate(pbar): 
                total_train_steps += 1
                # stop when the maximum number of training steps is reached
                if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                    max_steps_reached = True
                    accelerator.print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)

                y = batch.data.pop('y')

                with autocast():
                    # get the
                    output = model(**batch)
                    logits = output.logits
                    loss_con = output.loss

                num_token = logits[:,-1,:] # get the logit of the confidence token
                del logits
                scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(y.shape[0], 101).to(model.device)

                if train_config.loss_type == 'brier':
                    num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                    num_conf = F.softmax(num_conf, dim=1)
                    # compute the loss
                    y_expanded = y.expand(y.shape[0], 101)
                    squared_differences = (y_expanded - scores) ** 2
                    loss_cal = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
                elif train_config.loss_type == 'sot':
                    norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                    smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                    smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                    loss_cal = -torch.sum(norm_logit * smoothed, dim=1).mean()
                if train_config.add_loss_con:
                    loss = loss_con + loss_cal
                else:
                    loss = loss_cal
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.detach().float()

                if wandb_run and accelerator.is_main_process:
                    wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                            'train/loss_con': loss_con.detach().float(),
                            'train/loss_cal': loss_cal.detach().float()
                        })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    torch.cuda.empty_cache()
 
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        avg_loss = total_loss / len(train_dataloader)
        avg_ppl = float(torch.exp(torch.tensor(avg_loss)))  # ppl = exp(loss)

        # Update the learning rate as needed
        lr_scheduler.step()

        if train_config.run_validation and epoch == (train_config.num_epochs - 1):
            # Evaluate on all 5 datasets
            total_eval_score = 0.0
            total_eval_loss = 0.0
            dataset_scores = {}
            original_dataset = train_config.dataset
            for dataset_name, eval_dataloader in eval_dataloaders_dict.items():
                # Store original dataset name for logging
                train_config.dataset = dataset_name
                
                accelerator.print(f"Evaluating on dataset: {dataset_name}")
                
                temp_wandb_run = None
                if accelerator.is_main_process and wandb_run:
                    temp_metrics = {}
                    def temp_log(metrics_dict, commit=True):
                        temp_metrics.update(metrics_dict)
                    
                    original_log = wandb_run.log
                    wandb_run.log = temp_log
                
                eval_ppl, eval_epoch_loss, _, _ = evaluation_chat(
                    model,
                    train_config,
                    eval_dataloader,
                    tokenizer,
                    accelerator,
                    wandb_run,
                )
                
                if accelerator.is_main_process and wandb_run:
                    wandb_run.log = original_log
                    
                    dataset_metrics = {}
                    for key, value in temp_metrics.items():
                        if key.startswith('eval/'):
                            new_key = f'eval/{dataset_name}/{key[5:]}'
                            dataset_metrics[new_key] = value
                        elif key.startswith('plots/'):
                            dataset_metrics[key] = value
                    
                    wandb_run.log(dataset_metrics)
                    
                    ece_score = temp_metrics.get('eval/ece', 0)
                    roc_auc_score = temp_metrics.get('eval/roc_auc', 0)
                    
                    dataset_score = 2 * roc_auc_score - ece_score
                    dataset_scores[dataset_name] = dataset_score
                    total_eval_score += dataset_score
                    
                    wandb_run.log({
                        f'eval/{dataset_name}/score': dataset_score,
                    })
                    
                    accelerator.print(f"Dataset {dataset_name} - Score: {dataset_score:.4f}, Loss: {eval_epoch_loss:.4f}")

            train_config.dataset = original_dataset
            
            # Calculate average scores and log them
            if accelerator.is_main_process and wandb_run:
                # Log aggregate metrics
                wandb_run.log({
                    'eval/summary/total_score': total_eval_score,
                })
                accelerator.print(f"Evaluation Total Score: {total_eval_score:.4f}")

            
            # Save model if validation performance improved
            if accelerator.is_main_process:
                if True:
                    if train_config.save_model:
                        if train_config.use_peft:
                            if hasattr(model, "module"):
                                model = model.module
                            if train_config.merge_peft and epoch == (train_config.num_epochs - 1):
                                model = model.to(dtype=torch.float32)
                                model = model.merge_and_unload()
                                model = model.to(dtype=torch.float16)
                                save_merged_checkpoint(model, tokenizer, train_config.output_dir)
                                accelerator.print(f"Merged modules are saved in {train_config.output_dir} directory")
                            else:
                                save_peft_checkpoint(model, train_config.output_dir)
                                accelerator.print(f"PEFT modules are saved in {train_config.output_dir} directory")
                        else:
                            save_model_checkpoint(model, train_config.output_dir)
                            accelerator.print(f"Model is saved in {train_config.output_dir} directory")

    results = None
    sys.exit(0)

    return results

def evaluation_chat(
    model,
    train_config,
    eval_dataloader,
    tokenizer,
    accelerator,
    wandb_run=None,
    original=False
):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()
    all_y = []
    eval_probs = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_loss_con = 0.0
    eval_loss_cal = 0.0
    total_eval_steps = 0
    # Get the ids of the numbers from 0 to 100
    numbers = [str(i) for i in range(101)]
    token_ids = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    num_indices = torch.tensor(token_ids).to(model.device)
    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 100,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True
    }
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            y = batch.data.pop('y')
            output = model(**batch)
            logits = output.logits
            loss_con = output.loss
            num_token = logits[:,-1,:] # get the logit of the confidence token
            del logits
            scores = torch.arange(0, 1.01, 0.01).view(1, 101).expand(y.shape[0], 101).to(model.device)

            if train_config.loss_type == 'brier':
                num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                num_conf = F.softmax(num_conf, dim=1)
                # compute the loss
                y_expanded = y.expand(y.shape[0], 101)
                squared_differences = (y_expanded - scores) ** 2
                loss_cal = torch.mean(torch.sum(num_conf * squared_differences, dim=1))
            elif train_config.loss_type == 'sot':
                norm_logit = torch.index_select(F.log_softmax(num_token, dim=1), 1, num_indices.squeeze(0))
                smoothed = y * scores * (2 - scores) + (1 - y) * (1 - scores) * (1 + scores)
                smoothed = smoothed / smoothed.sum(dim=1, keepdim=True)
                loss_cal = -torch.sum(norm_logit * smoothed, dim=1).mean()
            if train_config.save_metrics:
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))

            if train_config.add_loss_con:
                loss = loss_con + loss_cal
            else:
                loss = loss_cal
            
            eval_loss += loss.detach().float()
            eval_loss_con += loss_con.detach().float()
            eval_loss_cal += loss_cal.detach().float()
            probs = 0.01 * torch.argmax(torch.index_select(num_token, 1, num_indices.squeeze(0)), dim=1)
            probs, y = accelerator.gather_for_metrics((probs, y))
            eval_probs.extend(probs.detach().cpu().numpy().tolist())
            all_y.extend(y.squeeze(1).detach().cpu().numpy().tolist())

    total_num = len(all_y)
    
    gathered_all_y = all_y
    gathered_eval_probs = eval_probs

    eval_epoch_loss = eval_loss / len(eval_dataloader) / accelerator.num_processes
    eval_epoch_loss_con = eval_loss_con / len(eval_dataloader) / accelerator.num_processes
    eval_epoch_loss_cal = eval_loss_cal / len(eval_dataloader) / accelerator.num_processes
    eval_ppl = torch.exp(eval_epoch_loss)

    if accelerator.is_main_process:
        all_gathered_y = gathered_all_y
        all_gathered_probs = gathered_eval_probs
        
        accelerator.print(f"====== {train_config.dataset}")
        accelerator.print(f"Number: {total_num}")
        val_metrics = compute_conf_metrics(all_gathered_y, all_gathered_probs, total_num)
        ece_score = val_metrics['ece']
        roc_auc_score = val_metrics['auroc']

        if train_config.use_wandb:
            plot_confidence_histogram(
                all_gathered_y, all_gathered_probs, "evaluation", 
                val_metrics['acc'], val_metrics['auroc'], val_metrics['ece'], 
                wandb_run, original, train_config.dataset, use_annotation=True
            )
            plot_ece_diagram(
                all_gathered_y, all_gathered_probs, "evaluation", 
                wandb_run, original, train_config.dataset
            )
        
        accelerator.print(f" {eval_ppl=} {eval_epoch_loss=}")

        if wandb_run:
            wandb_run.log({
                'eval/perplexity': eval_ppl,
                'eval/loss': eval_epoch_loss,
                'eval/loss_con': eval_epoch_loss_con,
                'eval/loss_cal': eval_epoch_loss_cal,
                'eval/ece': ece_score,
                'eval/roc_auc': roc_auc_score,
            }, commit=False)
    else:
        ece_score = 0
        roc_auc_score = 0

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

def test_vllm(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    all_y = []
    test_probs = []
    test_probs_stage1 = []
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=train_config.seed,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    if train_config.test_linguistic:
        responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace_3level(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    elif train_config.test_correct:
        responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace_correct(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    else:
        responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    if wandb_run:
        if original == True:
            wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})
        else:
            wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned": wan_table})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    if wandb_run:
        wandb_run.log({
                        f'test/number_{train_config.dataset}': number,
                        f'test/acc_{train_config.dataset}': val_metrics['acc'],
                        f'test/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'test/ece_{train_config.dataset}': ece_score,
                        f'test/auroc_{train_config.dataset}': roc_auc_score,
                        f'test/score_{train_config.dataset}': score,
                    }, commit=False)

    return ece_score, roc_auc_score, val_metrics['acc2']


def test_gpt(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader using OpenAI API

    Args:
        train_config: The training configuration
        test_dataset: The dataset containing the test data
        tokenizer: The tokenizer used to process prompts
        wandb_run: The wandb run object for logging
        original: Whether to use the original model or the fine-tuned model

    Returns: ece_score, roc_auc_score, accuracy
    """
    all_y = []
    test_probs = []
    test_probs_stage1 = []
    original = True
    # Initialize OpenAI client
    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
            api_version=os.environ['OPENAI_API_VERSION'],
            azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
        )
    except:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ['OPENAI_API_KEY'],
        )
    # Create a wandb table for logging
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    
    # Process prompts
    prompts = [json.loads(item) for item in test_dataset["prompt"]]

    # Generate responses using OpenAI API
    outputs = []
    for prompt in prompts:
        try:
            response = client.chat.completions.create(
                model=os.environ['OPENAI_DEPLOYMENT_NAME'],
                messages=prompt,
                temperature=train_config.temperature,
                max_tokens=400,
                top_p=1.0,
                n=1
            )
            outputs.append(response.choices[0].message.content)
        except Exception as e:
            print(f"Error generating response: {e}")
            outputs.append("")
    
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace_gpt(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=False)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        

    # Log to wandb
    if wandb_run:
        if original == True:
            wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})
        else:
            wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned": wan_table})

    # Compute metrics
    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    
    # Plot metrics if wandb is enabled
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original, train_config.dataset)

    # Calculate scores
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    # Log metrics to wandb
    if wandb_run:
        wandb_run.log({
                        f'gpt/number_{train_config.dataset}': number,
                        f'gpt/acc_{train_config.dataset}': val_metrics['acc'],
                        f'gpt/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'gpt/ece_{train_config.dataset}': ece_score,
                        f'gpt/auroc_{train_config.dataset}': roc_auc_score,
                        f'gpt/score_{train_config.dataset}': score,
                    }, commit=False)
    prompts2 = []
    for prompt, response in zip(prompts, responses):
        prompt[2]['content'] += (' ' + response)
        prompts2.append(prompt)
    
    original = False
    prompts2 = tokenizer.apply_chat_template(prompts2, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    outputs = llm.generate(prompts=prompts2, sampling_params=sampling_params)
    confidences = []
    import re
    for output in outputs:
        percent_str = output.outputs[0].text
        match = re.search(r"(\d+\.?\d*)%", percent_str)
        if match:
            percent = float(match.group(1)) / 100
            confidences.append(percent)
        else:
            confidences.append(0.0)
            print(f"Warning: Invalid confidence format: {percent_str}")
    val_metrics = compute_conf_metrics(y, confidences, len(prompts2))
    
    # Plot metrics if wandb is enabled
    if train_config.use_wandb:
        plot_confidence_histogram(y, confidences, "stage2", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, confidences, "stage2", wandb_run, original, train_config.dataset)

    # Calculate scores
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    # Log metrics to wandb
    if wandb_run:
        wandb_run.log({
                        f'test/number_{train_config.dataset}': number,
                        f'test/acc_{train_config.dataset}': val_metrics['acc'],
                        f'test/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'test/ece_{train_config.dataset}': ece_score,
                        f'test/auroc_{train_config.dataset}': roc_auc_score,
                        f'test/score_{train_config.dataset}': score,
                    }, commit=False)
    
    return responses



def test_reflection(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader using OpenAI API

    Args:
        train_config: The training configuration
        test_dataset: The dataset containing the test data
        tokenizer: The tokenizer used to process prompts
        wandb_run: The wandb run object for logging
        original: Whether to use the original model or the fine-tuned model

    Returns: ece_score, roc_auc_score, accuracy
    """
    reflection_prompt =  """For the question, response, and confidence, if the confidence is less than 50%, please revise your response and provide a better one. Otherwise, please repeat the response and the confidence.

            Here is the example:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was Percy Bysshe Shelley.
            Confidence: 40%
            If the confidence is less than 50%, analyze the answer and provide a better one. 
            Reflection: The response is less than 50%. 
            Response: The author of Paradise Lost wasn't Percy Bysshe Shelley, it was John Milton, who published the book in 1667.
            Confidence: 90%
            """
    original = True
    llm = LLM(
        model=train_config.model_name,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage1", wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    if wandb_run:
        wandb_run.log({
                        f'origin/number_{train_config.dataset}': number,
                        f'origin/acc_{train_config.dataset}': val_metrics['acc'],
                        f'origin/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'origin/ece_{train_config.dataset}': ece_score,
                        f'origin/auroc_{train_config.dataset}': roc_auc_score,
                        f'origin/score_{train_config.dataset}': score,
                    }, commit=False)
    del llm
    original = False
    prompts2 = []
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    for prompt, response, confidence in zip(prompts, out_response_cleans, out_confidences):
        if confidence < 0.5:
            prompt[0]['content'] = reflection_prompt
            prompt[1]['content'] += ('\nResponse:' + response + str(int(confidence * 100)) + '%\n' + 'The confidence is less than 50%, analyze the answer step by step and provide a better one.')
            prompt[2]['content'] = 'Reflection: The response is '
            prompts2.append(prompt)
        else:
            prompts2.append(prompt)
    
    wan_table2 = wandb.Table(columns=['response','confidence', 'y'])
    prompts2 = tokenizer.apply_chat_template(prompts2, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    llm = LLM(
        model=train_config.model_name if original else train_config.output_dir,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    outputs = llm.generate(prompts=prompts2, sampling_params=sampling_params)
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses, confidences_None, y_None):
        wan_table2.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    wandb_run.log({f"Testing_{train_config.dataset}/fine-tuned": wan_table2})

    number = len(y)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y, out_confidences, len(prompts)) 
    # Plot metrics if wandb is enabled
    if train_config.use_wandb:
        plot_confidence_histogram(y, out_confidences, "stage2", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, out_confidences, "stage2", wandb_run, original, train_config.dataset)

    # Calculate scores
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    # Log metrics to wandb
    if wandb_run:
        wandb_run.log({
                        f'test/number_{train_config.dataset}': number,
                        f'test/acc_{train_config.dataset}': val_metrics['acc'],
                        f'test/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'test/ece_{train_config.dataset}': ece_score,
                        f'test/auroc_{train_config.dataset}': roc_auc_score,
                        f'test/score_{train_config.dataset}': score,
                    }, commit=False)
    
    return responses


def test_reflection_gpt(train_config, test_dataset, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader using OpenAI API with different numbers of low-confidence samples
    being regenerated by GPT. Optimized to call the API only once for the maximum sample count.

    Args:
        train_config: The training configuration
        test_dataset: The dataset containing the test data
        tokenizer: The tokenizer used to process prompts
        wandb_run: The wandb run object for logging
        original: Whether to use the original model or the fine-tuned model

    Returns: ece_score, roc_auc_score, accuracy
    """
    reflection_prompt =  """For the question, response, and confidence, if the confidence is less than 50%, please revise your response and provide a better one. Otherwise, please repeat the response and the confidence.

            Here is the example:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was Percy Bysshe Shelley.
            Confidence: 40%
            If the confidence is less than 50%, analyze the answer and provide a better one. 
            Reflection: The response is less than 50%. 
            Response: The author of Paradise Lost wasn't Percy Bysshe Shelley, it was John Milton, who published the book in 1667.
            Confidence: 90%
            """
    original = True
    llm = LLM(
        model=train_config.model_name,
        tensor_parallel_size=1,
        dtype="float16",
        seed=42,
        disable_log_stats=True,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(
                                     n=1,
                                     temperature=train_config.temperature,
                                    top_k= -1,
                                    top_p=1.0,
                                     max_tokens=400)

    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses_stage1, out_response_cleans_stage1, questions_stage1, out_confidences_stage1, y_stage1, y_None_stage1, confidences_None_stage1, correct_answer_cleans_stage1 = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    for response, confidence, y_item in zip(responses_stage1, confidences_None_stage1, y_None_stage1):
        wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score for stage 1
    wandb_run.log({f"Testing_{train_config.dataset}/original": wan_table})

    number = len(y_stage1)
    print(f"Number: {number}")
    val_metrics = compute_conf_metrics(y_stage1, out_confidences_stage1, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y_stage1, out_confidences_stage1, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y_stage1, out_confidences_stage1, "stage1", wandb_run, original, train_config.dataset)

    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']
    score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

    if wandb_run:
        wandb_run.log({
                        f'origin/number_{train_config.dataset}': number,
                        f'origin/acc_{train_config.dataset}': val_metrics['acc'],
                        f'origin/acc2_{train_config.dataset}': val_metrics['acc2'],
                        f'origin/ece_{train_config.dataset}': ece_score,
                        f'origin/auroc_{train_config.dataset}': roc_auc_score,
                        f'origin/score_{train_config.dataset}': score,
                    }, commit=False)
    del llm
    
    confidence_with_indices = [(conf, i) for i, conf in enumerate(out_confidences_stage1)]
    confidence_with_indices.sort(key=lambda x: x[0])
    
    sample_counts = [10, 50, 80, 100, 200, 300, 400]
    max_count = min(max(sample_counts), len(confidence_with_indices))
    original = False
    
    metrics_table = wandb.Table(columns=["sample_count", "acc", "acc2", "ece", "auroc", "score"])
    
    print(f"\n===== Generating responses for {max_count} lowest confidence samples =====")
    
    lowest_conf_indices = [idx for _, idx in confidence_with_indices[:max_count]]
    
    prompts_low_conf = []
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    for idx in lowest_conf_indices:
        prompt = prompts[idx].copy()
        prompts_low_conf.append(prompt)
    
    gpt_responses = []
    gpt_confidences = []
    gpt_y = []
    
    if prompts_low_conf:
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
                api_version=os.environ['OPENAI_API_VERSION'],
                azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
            )
        except:
            from openai import OpenAI
            client = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
            )
        
        outputs_low_conf = []
        for i, prompt in enumerate(prompts_low_conf):
            try:
                print(f"Generating response for sample {i+1}/{len(prompts_low_conf)}")
                response = client.chat.completions.create(
                    model=os.environ.get('OPENAI_DEPLOYMENT_NAME'),
                    messages=prompt,
                    temperature=train_config.temperature,
                    max_tokens=400,
                    top_p=1.0,
                    n=1
                )
                outputs_low_conf.append(response.choices[0].message.content)
            except Exception as e:
                print(f"Error generating response with GPT API: {e}")
                outputs_low_conf.append("")
        
        questions_low_conf = [test_dataset['question'][idx] for idx in lowest_conf_indices]
        correct_answers_low_conf = [test_dataset['correct_answer'][idx] for idx in lowest_conf_indices]
        
        responses_stage2, out_response_cleans_stage2, questions_stage2, out_confidences_stage2, y_stage2, y_None_stage2, confidences_None_stage2, correct_answer_cleans_stage2 = confidence_replace_gpt(questions_low_conf, outputs_low_conf, correct_answers_low_conf, dataset_name=train_config.dataset, vllm=False)
        
        for i in range(len(responses_stage2)):
            gpt_responses.append(responses_stage2[i])
            gpt_confidences.append(out_confidences_stage2[i])
            gpt_y.append(y_stage2[i])
    
    for count in sample_counts:
        actual_count = min(count, max_count, len(confidence_with_indices))
        print(f"\n===== Evaluating with {actual_count} lowest confidence samples =====")
        
        current_lowest_indices = [idx for _, idx in confidence_with_indices[:actual_count]]
        other_indices = [idx for idx in range(len(out_confidences_stage1)) if idx not in lowest_conf_indices[:actual_count]]
        
        wan_table2 = wandb.Table(columns=['response', 'confidence', 'y', 'source'])
        
        final_responses = []
        final_confidences = []
        final_y = []
        
        for idx in other_indices:
            final_responses.append(responses_stage1[idx])
            final_confidences.append(out_confidences_stage1[idx])
            final_y.append(y_stage1[idx])
            # wan_table2.add_data(responses_stage1[idx], confidences_None_stage1[idx], y_None_stage1[idx], "stage1-original")
        
        for i in range(min(actual_count, len(gpt_responses))):
            final_responses.append(gpt_responses[i])
            final_confidences.append(gpt_confidences[i])
            final_y.append(gpt_y[i])

        val_metrics = compute_conf_metrics(final_y, final_confidences, len(final_y))
        
        if train_config.use_wandb:
            plot_confidence_histogram(final_y, final_confidences, f"combined_{actual_count}", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
            plot_ece_diagram(final_y, final_confidences, f"combined_{actual_count}", wandb_run, original, train_config.dataset)

        ece_score = val_metrics['ece']
        roc_auc_score = val_metrics['auroc']
        score = 3 * val_metrics['acc2'] + 2 * roc_auc_score - ece_score

        metrics_dict = {
            f'test_{actual_count}/number_{train_config.dataset}': len(final_y),
            f'test_{actual_count}/acc_{train_config.dataset}': val_metrics['acc'],
            f'test_{actual_count}/acc2_{train_config.dataset}': val_metrics['acc2'],
            f'test_{actual_count}/ece_{train_config.dataset}': ece_score,
            f'test_{actual_count}/auroc_{train_config.dataset}': roc_auc_score,
            f'test_{actual_count}/score_{train_config.dataset}': score,
            f'test_{actual_count}/low_conf_count': actual_count,
            f'test_{actual_count}/high_conf_count': len(other_indices),
        }
        wandb_run.log(metrics_dict, commit=False)
        
        metrics_table.add_data(actual_count, val_metrics['acc'], val_metrics['acc2'], ece_score, roc_auc_score, score)
        
        print(f"Sample count: {actual_count}")
        print(f"Accuracy: {val_metrics['acc']:.4f}")
        print(f"Accuracy (correct with conf): {val_metrics['acc2']:.4f}")
        print(f"ECE: {ece_score:.4f}")
        print(f"AUROC: {roc_auc_score:.4f}")
        print(f"Score: {score:.4f}")
    
    wandb_run.log({f"metrics_summary_{train_config.dataset}": metrics_table})
    
    return final_responses


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False

def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")

def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")

def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")

def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()

def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()

def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")

def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and torch.version.cuda >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))

    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
