import jsonargparse
from tqdm import tqdm 
import torch
import json 
from collections import defaultdict
import pdb 
import re
import sys
import os
# from vllm import LLM, SamplingParams
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)


from trained_calibration.rl.dataset.dataset import get_dataset
from trained_calibration.rl.reward_model import RewardModel
from trained_calibration.rl.dataset.postprocess import postprocess_answers, postprocess_extract

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM

def main(args):

    if args.split is None:
        split = "train"
    else:
        split = args.split


    dataset = get_dataset(args.dataset)
    if args.limit is not None:
        dataset_to_run = dataset[split].select(range(args.limit))
    else: 
        dataset_to_run = dataset[split]

    dataloader = torch.utils.data.DataLoader(
            dataset_to_run,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
    a = 0
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    )

    generator = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        quantization_config=bnb_config,
        use_cache=False,
        attn_implementation="sdpa",
        device_map="auto" ,
        torch_dtype=torch.float16,
    )
    # generator = LLM(args.model_name)
    
    # generator = AutoModelForCausalLM.from_pretrained(args.model_name).half()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
    device = args.model_device_map['main']
    # device = f"cuda:{args.model_device_list}"

    pad_token = None
    if "llama" in args.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_token = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        pad_token = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    # generator.to(device)

    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    )
    # reward_model_names = args.reward_model_names.split(",")
    # reward_model_devices = args.reward_model_devices.split(",")

    # reward_models = [RewardModel(model_name, args.model_device_map[f'reward{i}'], quantization_config=bnb_config) for i, model_name in enumerate(args.reward_model_names)]

    generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 80,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.1,
        "do_sample": True,
        "pad_token_id": pad_token,
    }
    #sampling_params = SamplingParams(temperature=0.7, top_p=1.0)

    with open(args.output_dir, "w") as f1:
        # id=0
        for batch in tqdm(dataloader): 
            try:
                # print(id)
                # if id <=284:
                #     id += 1
                #     continue
                responses_by_example = []
                prompts = [json.loads(item) for item in batch["generator_prompt"]]
                query_tensors = tokenizer.apply_chat_template(prompts, tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).to(generator.device)
                
                for i in range(args.n_generations):
                    with torch.no_grad():
                        response_tensors = generator.generate(query_tensors, **generation_kwargs) 
                    batch_responses = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
                    
                    questions, out_responses, confidences, correct_answers_final = postprocess_extract(prompts, batch_responses, batch['correct_answer'], args.dataset)

                    for query_ids in range(len(questions)):
                        responses_by_example.append({
                                                                "question": questions[query_ids],
                                                                "response_clean": out_responses[query_ids], 
                                                                "confidence": confidences[query_ids],
                                                                "correct_answer": json.loads(correct_answers_final[query_ids])})
                for item in responses_by_example:
                    json_line = json.dumps(item)  # 将每个 JSON 对象转换为字符串
                    f1.write(json_line + "\n")    # 写入文件，并换行
                f1.flush()
               
            except RuntimeError as e:
                print(f"Batch failed with exception: {e}")
                continue
            # except RuntimeError:
            #     continue
        

def extract_response(response):
    try:
        prompt, response = re.split("([Aa]nswer:)", response)
    except ValueError:
        return None
    # TODO (elias): remove incomplete sentences 
    return response.strip()

if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--cfg", action=jsonargparse.ActionConfigFile, help="path to config file")
    parser.add_argument("--model_name", type=str, default="/home/lyb/workspace/meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--reward_model_names", type=list, default=None, help="list of reward model names") 
    parser.add_argument("--model_device_map", type=dict, default="0", help="dict specifying which devices have which model")
    parser.add_argument("--dataset", type=str, default="trivia_qa")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_generations", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    args.model_name= "/home/lyb/workspace/meta-llama/Llama-3.1-8B-Instruct" # TODO
    args.reward_model_names = [ "/home/lyb/workspace/meta-llama/Llama-3.1-8B-Instruct"]
    args.output_dir = "/home/lyb/workspace/Uncertainty_ft/dataset/trivia_qa/tqa_val_multi.jsonl"
    args.split = 'validation'
    args.batch_size=24
    args.n_generations=10
    main(args)