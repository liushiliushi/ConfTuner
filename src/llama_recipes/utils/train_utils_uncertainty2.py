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
    """
    兼容 accelerate 的简易 profiler/flop 计数器上下文管理。
    """
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
        # 如果想在多卡时区分 rank，可用 accelerator.process_index
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
    # 根据是否使用混合精度来设置 autocast
    if train_config.use_fp16 and torch.cuda.is_available():
        autocast = torch.cuda.amp.autocast
    else:
        autocast = contextlib.nullcontext  # 使用 nullcontext 作为默认值

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
    # Get the ids for high/middle/low confidence levels
    confidence_levels = ["high", "middle", "low"]
    token_ids = [tokenizer.encode(level, add_special_tokens=False)[0] for level in confidence_levels]
    print("----------------------------")
    print(token_ids)
    conf_indices = torch.tensor(token_ids).to(model.device)
    # Map confidence levels to actual confidence values
    conf_values = {"high": 5/6, "middle": 3/6, "low": 1/6}
    conf_scores = torch.tensor([conf_values["high"], conf_values["middle"], conf_values["low"]]).to(model.device)
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
                # TODO
                print(batch['input_ids'].shape[1])
                if batch['input_ids'].shape[1] > 350:
                    continue
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

                conf_token = logits[:,-1,:] # get the logit of the confidence token
                del logits
                
                if train_config.loss_type == 'brier':
                    # Extract logits for high/middle/low tokens
                    conf_logits = torch.index_select(conf_token, 1, conf_indices) # take out the logits of high/middle/low
                    conf_probs = F.softmax(conf_logits, dim=1)
                    
                    # Calculate expected confidence value based on probabilities
                    expected_conf = torch.sum(conf_probs * conf_scores.view(1, 3), dim=1, keepdim=True)
                    
                    # Compute the Brier score: (y - conf)^2
                    loss_cal = torch.mean((y - expected_conf) ** 2)
                elif train_config.loss_type == 'sot':
                    # Get log probabilities for the confidence tokens
                    norm_logit = torch.index_select(F.log_softmax(conf_token, dim=1), 1, conf_indices)
                    
                    # Create target distribution based on ground truth
                    # For correct answers (y=1), put higher probability on high confidence
                    # For incorrect answers (y=0), put higher probability on low confidence
                    target_probs = torch.zeros((y.shape[0], 3), device=model.device)
                    
                    # Create smoothed target distribution based on correctness
                    for i in range(y.shape[0]):
                        if y[i] > 0.5:  # Correct answer
                            target_probs[i] = torch.tensor([0.7, 0.2, 0.1], device=model.device)  # high, middle, low
                        else:  # Incorrect answer
                            target_probs[i] = torch.tensor([0.1, 0.2, 0.7], device=model.device)  # high, middle, low
                    
                    # Cross-entropy loss
                    loss_cal = -torch.sum(norm_logit * target_probs, dim=1).mean()
                
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

        if train_config.run_validation:
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

    # accelerator.wait_for_everyone()
    # model.cpu() 
    # del model
    # torch.cuda.empty_cache() 
    # rank = accelerator.process_index
    # print(f"[rank={rank}] Before load model")
    # if accelerator.is_main_process:
    # accelerator.print("==============finetuned test2stage================")
        
    # test_ece, test_auroc = test_vllm(train_config, test_dataloader, tokenizer, wandb_run, original=False)
    # rank = accelerator.process_index
    # print(f"[rank={rank}] After load model")
    # accelerator.wait_for_everyone()
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
    # Get the ids for high/middle/low confidence levels
    confidence_levels = ["high", "middle", "low"]
    token_ids = [tokenizer.encode(level, add_special_tokens=False)[0] for level in confidence_levels]
    conf_indices = torch.tensor(token_ids).to(model.device)
    # Map confidence levels to actual confidence values
    conf_values = {"high": 5/6, "middle": 3/6, "low": 1/6}
    conf_scores = torch.tensor([conf_values["high"], conf_values["middle"], conf_values["low"]]).to(model.device)
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
            conf_token = logits[:,-1,:] # get the logit of the confidence token
            del logits
            
            if train_config.loss_type == 'brier':
                # Extract logits for high/middle/low tokens
                conf_logits = torch.index_select(conf_token, 1, conf_indices) # take out the logits of high/middle/low
                conf_probs = F.softmax(conf_logits, dim=1)
                
                # Calculate expected confidence value based on probabilities
                expected_conf = torch.sum(conf_probs * conf_scores.view(1, 3), dim=1, keepdim=True)
                
                # Compute the Brier score: (y - conf)^2
                loss_cal = torch.mean((y - expected_conf) ** 2)
            elif train_config.loss_type == 'sot':
                # Get log probabilities for the confidence tokens
                norm_logit = torch.index_select(F.log_softmax(conf_token, dim=1), 1, conf_indices)
                
                # Create target distribution based on ground truth
                # For correct answers (y=1), put higher probability on high confidence
                # For incorrect answers (y=0), put higher probability on low confidence
                target_probs = torch.zeros((y.shape[0], 3), device=model.device)
                
                # Create smoothed target distribution based on correctness
                for i in range(y.shape[0]):
                    if y[i] > 0.5:  # Correct answer
                        target_probs[i] = torch.tensor([0.7, 0.2, 0.1], device=model.device)  # high, middle, low
                    else:  # Incorrect answer
                        target_probs[i] = torch.tensor([0.1, 0.2, 0.7], device=model.device)  # high, middle, low
                
                # Cross-entropy loss
                loss_cal = -torch.sum(norm_logit * target_probs, dim=1).mean()
            
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
            
            # Calculate confidence scores from the high/middle/low tokens
            conf_logits = torch.index_select(conf_token, 1, conf_indices)
            conf_probs = F.softmax(conf_logits, dim=1)
            
            # Calculate expected confidence from probabilities
            confidence_values = torch.sum(conf_probs * conf_scores.view(1, 3), dim=1)
            eval_probs.extend(confidence_values.detach().cpu().numpy().tolist())
            all_y.extend(y.squeeze(1).detach().cpu().numpy().tolist())

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    number = len(all_y)
    val_metrics = compute_conf_metrics(all_y, eval_probs, number)
    # if train_config.use_wandb and accelerator.is_main_process:
    #     plot_confidence_histogram(all_y, eval_probs, "evaluation", val_metrics['acc'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
    #     plot_ece_diagram(all_y, eval_probs, "evaluation", wandb_run, original, train_config.dataset)
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_loss_con = eval_loss_con / len(eval_dataloader)
    eval_epoch_loss_cal = eval_loss_cal / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if accelerator.is_main_process:
        accelerator.print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run and accelerator.is_main_process:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                        'eval/loss_con': eval_epoch_loss_con,
                        'eval/loss_cal': eval_epoch_loss_cal,
                        'eval/ece': ece_score,
                        'eval/roc_auc': roc_auc_score,
                    }, commit=False)
    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

def test_2stage(model, train_config, test_dataloader, local_rank, tokenizer, wandb_run, original=False):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    # llm = LLM(model=train_config.output_dir, max_model_len=93968)
    
    all_y = []
    test_probs = []
    test_probs_stage1 = []

    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 200, 
        "top_k": 0.0,
        "top_p": 1.0,
        "repetition_penalty": train_config.repetition_penalty,
        "temperature": train_config.temperature,
        "do_sample": train_config.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "output_scores": True,
        "return_dict_in_generate": True,
        # "pad_token_id": tokenizer.pad_token_id
    }
    # Get the ids for high/middle/low confidence levels
    confidence_levels = ["high", "middle", "low"]
    token_ids = [tokenizer.encode(level, add_special_tokens=False)[0] for level in confidence_levels]
    conf_indices = torch.tensor(token_ids).to(model.device)
    # Map confidence levels to actual confidence values
    conf_values = {"high": 5/6, "middle": 3/6, "low": 1/6}
    conf_scores = torch.tensor([conf_values["high"], conf_values["middle"], conf_values["low"]]).to(model.device)
    with MemoryTrace() as memtrace:
        id = 0
        wan_table = wandb.Table(columns=['response','confidence', 'y'])
        for step, batch in enumerate(tqdm(test_dataloader,colour="green", desc="testing Epoch", dynamic_ncols=False)):
            
            prompts = [json.loads(item) for item in batch["prompt"]]
            query_tensors = tokenizer.apply_chat_template(prompts, tokenize=True, padding="longest", padding_side='left', truncation=True, return_dict=True, return_tensors="pt", continue_final_message=True).to(model.device)
            # stop when the maximum number of eval steps is reached
            

            with torch.no_grad():
                output = model.generate(**query_tensors, **generation_kwargs, ) 
                # output = model(**query_tensors, **generation_kwargs)
                responses = output.sequences
                batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in responses]
                prompts_new, out_responses, questions, confidence_stage1, y, y_unfiltered, confidence_unfiltered,_ = confidence_replace(prompts, batch_responses, batch['correct_answer'], train_config.dataset)
                if len(prompts_new) == 0:
                    continue
                y = torch.tensor(y).view(-1, 1).to(model.device)

                # include attention mask
                query_tensors_new = tokenizer.apply_chat_template(prompts_new, tokenize=True, padding="longest", padding_side='left', truncation=True, return_dict=True, return_tensors="pt", continue_final_message=True).to(model.device)
                white_spaces = torch.full((len(prompts_new), 1), 220).to(model.device)
                white_attention = torch.full((len(prompts_new), 1), 1).to(model.device)
                query_tensors_new['input_ids'] = torch.cat([query_tensors_new['input_ids'], white_spaces], dim=1).to(model.device)
                query_tensors_new['attention_mask'] = torch.cat([query_tensors_new['attention_mask'], white_attention], dim=1).to(model.device)
                logits = model(**query_tensors_new).logits

                # exclude attention mask
                # query_tensors_new = tokenizer.apply_chat_template(prompts_new, tokenize=True, padding="longest", padding_side='left', truncation=True, return_tensors="pt", continue_final_message=True).to(model.device)
                # white_spaces = torch.full((len(prompts_new), 1), 220).to(model.device)
                # query_tensors_new = torch.cat([query_tensors_new, white_spaces], dim=1).to(model.device)
                # logits = model(query_tensors_new).logits

            conf_token = logits[:,-1,:]
            
            # Calculate confidence scores from the high/middle/low tokens
            conf_logits = torch.index_select(conf_token, 1, conf_indices)
            conf_probs = F.softmax(conf_logits, dim=1)
            
            # Calculate expected confidence from probabilities
            confidence_values = torch.sum(conf_probs * conf_scores.view(1, 3), dim=1)
            test_probs.extend(confidence_values.detach().cpu().numpy().tolist())
            test_probs_stage1.extend(confidence_stage1)
            all_y.extend(y.squeeze(1).detach().cpu().numpy().tolist())
            for response, confidence, y_item in zip(batch_responses, confidence_unfiltered, y_unfiltered):
                wan_table.add_data(response, confidence, y_item)        

    # Compute ECE and ROC-AUC score given all_y and eval_probs
    if wandb_run:
        if original == True:
            wandb_run.log({"Testing_samples/original": wan_table})
        else:
            wandb_run.log({"Testing_samples/fine-tuned": wan_table})

    number = len(all_y)
    print(f"Number: {number}")
    print("Stage1:")
    val_metrics_stage1 = compute_conf_metrics(all_y, test_probs_stage1)
    if train_config.use_wandb:
        plot_confidence_histogram(all_y, test_probs_stage1, "stage1", val_metrics_stage1['acc'], val_metrics_stage1['auroc'], val_metrics_stage1['ece'], wandb_run, original, use_annotation=True)
        plot_ece_diagram(all_y, test_probs_stage1, "stage1", wandb_run, original)

    print("Stage2:")
    val_metrics = compute_conf_metrics(all_y, test_probs)
    if train_config.use_wandb:
        plot_confidence_histogram(all_y, test_probs, "stage2", val_metrics['acc'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, use_annotation=True)
        plot_ece_diagram(all_y, test_probs, "stage2", wandb_run,original)
    
    ece_score = val_metrics['ece']
    roc_auc_score = val_metrics['auroc']

    if wandb_run:
        wandb_run.log({
                        'test/ece_stage1': ece_score,
                        'test/roc_auc_stage1': roc_auc_score,
                        'test/ece_stage2': ece_score,
                        'test/roc_auc_stage2': roc_auc_score,
                    }, commit=False)

    return ece_score, roc_auc_score

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

    # Get the ids for high/middle/low confidence levels
    confidence_levels = ["high", "middle", "low"]
    token_ids = [tokenizer.encode(level, add_special_tokens=False)[0] for level in confidence_levels]
    conf_indices = torch.tensor(token_ids)
    # Map confidence levels to actual confidence values
    conf_values = {"high": 5/6, "middle": 3/6, "low": 1/6}
    
    wan_table = wandb.Table(columns=['response','confidence', 'y'])
    prompts = [json.loads(item) for item in test_dataset["prompt"]]
    prompts = tokenizer.apply_chat_template(prompts, tokenize=False, padding="longest", truncation=True, return_tensors="pt",  continue_final_message=True)
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans = confidence_replace(test_dataset['question'], outputs, test_dataset['correct_answer'], dataset_name=train_config.dataset,vllm=True)
    
    # Convert string confidence levels (high/middle/low) to numerical values
    numerical_confidences = []
    for conf in out_confidences:
        if isinstance(conf, str):
            conf = conf.lower()
            if conf == "high":
                numerical_confidences.append(conf_values["high"])
            elif conf == "middle":
                numerical_confidences.append(conf_values["middle"])
            elif conf == "low":
                numerical_confidences.append(conf_values["low"])
            else:
                # Default to middle if the confidence string is unrecognized
                numerical_confidences.append(conf_values["middle"])
        else:
            # If it's already a number, use it directly (for backward compatibility)
            numerical_confidences.append(conf)
    
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
    val_metrics = compute_conf_metrics(y, numerical_confidences, len(prompts))
    if train_config.use_wandb:
        plot_confidence_histogram(y, numerical_confidences, "stage1", val_metrics['acc2'], val_metrics['auroc'], val_metrics['ece'], wandb_run, original, train_config.dataset, use_annotation=True)
        plot_ece_diagram(y, numerical_confidences, "stage1", wandb_run, original, train_config.dataset)

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
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
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
