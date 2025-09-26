# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
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
from llama_recipes.utils.postprocess import postprocess_extract, confidence_replace
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
    numbers = [str(i) for i in range(10)]
    token_ids1 = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    token_ids2 = [tokenizer.encode('0', add_special_tokens=False)[0], tokenizer.eos_token_id]
    print("----------------------------")
    print(token_ids1)
    num_indices1 = torch.tensor(token_ids1).to(model.device)
    num_indices2 = torch.tensor(token_ids2).to(model.device)
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
                    output1 = model(**batch)
                    logits1 = output1.logits
                    loss_con = output1.loss

                num_token1 = logits1[:,-1,:] # get the logit of the confidence token
                del logits1
                # scores = torch.arange(0, 1, 0.1).view(1, 10).expand(y.shape[0], 10).to(model.device)
                scores = torch.linspace(0, 1, 10).view(1, 10).expand(y.shape[0], 10).to(model.device)

                if train_config.loss_type == 'brier':
                    num_conf1 = torch.index_select(num_token1, 1, num_indices1.squeeze(0)) # take out the logit of 0-9+1
                    num_conf1 = F.softmax(num_conf1, dim=1)
                    y_expanded = y.expand(y.shape[0], 10)
                    squared_differences = (y_expanded - scores) ** 2
                    loss_cal = torch.mean(torch.sum(num_conf1 * squared_differences, dim=1))
                elif train_config.loss_type == 'sot':
                    norm_logit = torch.index_select(F.log_softmax(num_token1, dim=1), 1, num_indices1.squeeze(0))
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

                    # torch.cuda.empty_cache()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        avg_loss = total_loss / len(train_dataloader)
        avg_ppl = float(torch.exp(torch.tensor(avg_loss)))  # ppl = exp(loss)

        # Update the learning rate as needed
        lr_scheduler.step()
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, _, _ = evaluation_chat(
                model,
                train_config,
                eval_dataloader,
                tokenizer,
                accelerator,
                wandb_run,
            )
            if True:
                if accelerator.is_main_process:
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
                            sys.exit(0)
                        else:
                            save_peft_checkpoint(model, train_config.output_dir)
                            accelerator.print(f"PEFT modules are saved in {train_config.output_dir} directory")
                    else:
                        save_model_checkpoint(model, train_config.output_dir)
                        accelerator.print(f"Model is saved in {train_config.output_dir} directory")
    results = None
   

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
    numbers = [str(i) for i in range(10)]
    token_ids1 = [tokenizer.encode(number, add_special_tokens=False)[0] for number in numbers]
    token_ids2 = [tokenizer.encode('0', add_special_tokens=False)[0], tokenizer.eos_token_id]
    print("----------------------------")
    print(token_ids1)
    num_indices1 = torch.tensor(token_ids1).to(model.device)
    num_indices2 = torch.tensor(token_ids2).to(model.device)
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
                    output1 = model(**batch)
                    logits1 = output1.logits
                    loss_con = output1.loss

                num_token1 = logits1[:,-1,:] # get the logit of the confidence token
                del logits1
                scores = torch.arange(0, 1, 0.1).view(1, 10).expand(y.shape[0], 10).to(model.device)

                if train_config.loss_type == 'brier':
                    num_conf1 = torch.index_select(num_token1, 1, num_indices1.squeeze(0)) # take out the logit of 0-9+1
                    num_conf1 = F.softmax(num_conf1, dim=1)
                    y_expanded = y.expand(y.shape[0], 10)
                    squared_differences = (y_expanded - scores) ** 2
                    loss_cal = torch.mean(torch.sum(num_conf1 * squared_differences, dim=1))
                elif train_config.loss_type == 'sot':
                    norm_logit = torch.index_select(F.log_softmax(num_token1, dim=1), 1, num_indices1.squeeze(0))
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

                    # torch.cuda.empty_cache()

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
            if accelerator.is_main_process:
                # Log aggregate metrics
                if wandb_run:
                    wandb_run.log({
                        'eval/summary/total_score': total_eval_score,
                    })
                    accelerator.print(f"Evaluation Total Score: {total_eval_score:.4f}")

            

                if True:
                    if train_config.save_model:
                        if train_config.use_peft:
                            if hasattr(model, "module"):
                                model = model.module
                            if train_config.merge_peft and epoch == (train_config.num_epochs - 1):
                                clear_gpu_cache()
                                try:
                                    model = model.to(dtype=torch.float32)
                                    model = model.cpu()
                                    model = model.merge_and_unload()
                                    model = model.to(dtype=torch.float16)
                                    if torch.cuda.is_available():
                                        model = model.to("cuda")
                                    save_merged_checkpoint(model, tokenizer, train_config.output_dir)
                                    accelerator.print(f"Merged modules are saved in {train_config.output_dir} directory")
                                except Exception as e:
                                    accelerator.print(f"Error during model merge and save: {str(e)}")
                            else:
                                save_peft_checkpoint(model, train_config.output_dir)
                                accelerator.print(f"PEFT modules are saved in {train_config.output_dir} directory")
                        else:
                            save_model_checkpoint(model, train_config.output_dir)
                            accelerator.print(f"Model is saved in {train_config.output_dir} directory")
    
    results = None
   

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
    # Get the ids of the numbers from 0 to 9
    numbers = [str(i) for i in range(10)]
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
            scores = torch.arange(0, 1, 0.1).view(1, 10).expand(y.shape[0], 10).to(model.device)

            if train_config.loss_type == 'brier':
                num_conf = torch.index_select(num_token, 1, num_indices.squeeze(0)) # take out the logit of 0-100
                num_conf = F.softmax(num_conf, dim=1)
                # compute the loss
                y_expanded = y.expand(y.shape[0], 10)
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
            print(f"loss: {loss} loss_con: {loss_con} loss_cal: {loss_cal}") 
               
            eval_loss += loss.detach().float()
            eval_loss_con += loss_con.detach().float()
            eval_loss_cal += loss_cal.detach().float()
            probs = 0.1 * torch.argmax(torch.index_select(num_token, 1, num_indices.squeeze(0)), dim=1)

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
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
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
