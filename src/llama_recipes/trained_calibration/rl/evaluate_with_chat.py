import pdb
import json 
import argparse
import re
from tqdm import tqdm
from collections import defaultdict

from trained_calibration.rl.reward_model import RewardModel
from trained_calibration.rl.dataset.postprocess import postprocess_extract

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import random

def chatify(prompt, soi_token, eoi_token):
    prompt, question = re.split("Question:", prompt)
    question = re.sub("Response:", "", question).strip()
    prompt = f"{soi_token} {prompt.strip()} {eoi_token} \nQuestion: {question.strip()}\n{soi_token}Provide your response: {eoi_token}"
    return prompt 

def dechatify(output, soi_token, eoi_token):
    soi_token = re.escape(soi_token)
    eoi_token = re.escape(eoi_token)
    output = re.sub(f"{soi_token}Provide your response: {eoi_token}", "Response:", output)
    return output

def run_prompt(prompt, soi_token, eoi_token):
    prompt = chatify(prompt, soi_token, eoi_token)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    output = model.generate(input_ids, 
                            do_sample=True, 
                            max_new_tokens=80, 
                            temperature=0.7, 
                            top_k=0.0,
                            top_p=1.0,
                            pad_token_id = tokenizer.unk_token_id,
                            num_return_sequences=1)
    output_decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
    return [dechatify(x, soi_token, eoi_token) for x in output_decoded]
    # return output_decoded



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--trained_model", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--n_per_prompt", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.trained_model, cache_dir="/nas-ssd2/esteng/.cache")
    model = model.half()
    model = model.to("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.trained_model, cache_dir="/nas-ssd2/esteng/.cache")
    tokenizer.pad_token = tokenizer.unk_token

    soi_token = "[INST]"
    eoi_token = "[/INST]"

    reward_model = RewardModel(args.model, "cuda:1") 
    
    with open(args.data_path) as f1:
        data = [json.loads(line) for line in f1]

    data_by_prompt = defaultdict(list)
    for d in data:
        data_by_prompt[d['prompt']].append(d)

    i = 0

    true_positives = []
    false_positives = []
    false_negatives = []
    scores_for_auroc = []
    corrects_for_auroc = []
    nonanswers = []

    if args.limit == -1:
        limit = len(data_by_prompt)
    else:
        limit = args.limit
        data_by_prompt = dict(list(data_by_prompt.items())[:limit])

    with open(args.out_path, "w") as f1:

        for prompt, data in tqdm(data_by_prompt.items(), total=limit):
            prompts = []
            data_corrects = []
            # get first answer that's not NONE
            try:
                new_data = [d for d in data if d['answer'] != "NONE"]
                prompt = new_data[0]['prompt']
                data = new_data
            except IndexError:
                # pdb.set_trace()
                # back off to NONE if not possible 
                prompt = data[0]['prompt']

            correct_answers = data[0]['correct_answers']
            # print(f"Prompt: {prompt}")

            reference_outputs = []
            for d in data:
                try:
                    resp_only = re.split("Response:", d['response_clean'])[1].strip()
                except IndexError:
                    resp_only = d['response_clean']
                reference_outputs.append(resp_only)

            reference_answers = [d['answer'] for d in data]
            reference_probs = [d['mean_prob'] for d in data]
            reference_corrects = [d['correct'] for d in data]

            trained_outputs = []
            for i in range(len(data)):
                if i>= args.n_per_prompt:
                    break
                output = run_prompt(prompt, "[INST]", "[\INST]")[0]
                trained_outputs.append(output)
                prompts.append(prompt)
                data_corrects.append(correct_answers)

            out_responses, out_answers, __ = postprocess_extract(prompts, trained_outputs, reward_model.model, reward_model.tokenizer, "trivia_qa")

            out_responses_final = []
            for x in out_responses:
                try:
                    resp_only = re.split("Response:", x)[1].strip()
                except IndexError:
                    resp_only = x
                out_responses_final.append(resp_only)
            batch_size = 3
            if len(out_responses_final) == 0:
                out_responses_final = ["NONE" for x in prompts]
                out_answers = ["NONE" for x in prompts]

            all_scores = []
            all_corrects = []
            all_probs = []
            for i in range(0, len(out_responses_final), batch_size):
                prompt_batch = prompts[i:i+batch_size]

                for i, x in enumerate(prompt_batch):
                    x = dechatify(x, soi_token, eoi_token)

                    x = re.sub(re.escape(soi_token), "", x).strip()
                    x = re.sub(re.escape(eoi_token), "\n", x).strip()
                    prompt_batch[i] = x

                out_batch = out_responses_final[i:i+batch_size]
                answer_batch = out_answers[i:i+batch_size] 
                correct_batch = data_corrects[i:i+batch_size]
                # run reward model 
                scores, corrects, probs = reward_model.forward(prompt_batch, out_batch, answer_batch, correct_batch )
                probs = [x.item() for x in probs]
                scores = [x.item() for x in scores]
                all_scores.extend(scores)
                all_corrects.extend(corrects)
                all_probs.extend(probs)

                accept = [1 if p > args.threshold else 0 for p in all_probs]

                if accept == 1 and all_corrects == 1:
                    true_positives.append(1) 
                elif accept == 1 and all_corrects == 0:
                    false_positives.append(1)
                elif accept == 0 and all_corrects == 1:
                    false_negatives.append(1)
                else:
                    pass
                
            for i in range(len(prompts)):
                out_data = {"prompt": prompts[i],
                            "trained_output": out_responses_final[i],
                            "trained_answer": out_answers[i],
                            "correct_answers": data_corrects[i],
                            "trained_prob": all_probs[i],
                            "trained_correct": all_corrects[i],
                            "reference_output": reference_outputs[i],
                            "reference_prob": reference_probs[i],
                            "reference_answer": reference_answers[i],
                            "reference_correct": reference_corrects[i]}
                
                f1.write(json.dumps(out_data) + "\n")

                
            
