
import copy
import datasets
import os
import json
import os
import re
import torch
from datasets import concatenate_datasets

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
system_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Then please provide the final answer of yes or no. If no answer is present, please write "NONE".
            Finally, please provide your confidence (0%-100%) to your answer.

            Here are some examples:

            Question: Can plants survive without sunlight?
            Response: No. Photosynthesis depends on sunlight.
            Final answer: No.
            Confidence: 95%

            Question: Do all mammals lay eggs?
            Response: No. Only monotremes (e.g., platypus) do.
            Final answer: No.
            Confidence: 97%

            Question: Is water a good conductor of electricity?
            Response: No. Pure water lacks conductive ions.
            Final answer: No.
            Confidence: 90%"""
system_prompt_linguistic = """You will be asked trivia questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Then please provide the final answer of yes or no. If no answer is present, please write "NONE".
            Assess your confidence level based on:
                - High (66%-100%): Certain of correctness with logical reasoning
                - Medium (33%-66%): Partially confident but some uncertainty
                - Low (0%-33%): Suspect potential errors in calculation/logic

            Here are some examples:

            Question: Can plants survive without sunlight?
            Response: No. Photosynthesis depends on sunlight.
            Final answer: No.
            Confidence: high

            Question: Do all mammals lay eggs?
            Response: No. Only monotremes (e.g., platypus) do.
            Final answer: No.
            Confidence: high

            Question: Is water a good conductor of electricity?
            Response: No. Pure water lacks conductive ions.
            Final answer: No.
            Confidence: high"""

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def get_strategyqa(tokenizer, split, train_config, on_policy=False):
    if train_config.test_linguistic:
        system_prompt = system_prompt_linguistic
    if split == 'train':
        print("Error! Strategy QA is not used for training.")
    else:
        if train_config.train_gpt:
            path = '../dataset/StrategyQA/validation_gpt_temp=0.jsonl'
        else:
            path = '../dataset/StrategyQA/task.jsonl'
        dataset = datasets.load_dataset('json', data_files=path, split='train')

    def apply_prompt_template(sample):
        global system_prompt
        if train_config.test_linguistic:
            system_prompt = system_prompt_linguistic
        if train_config.train_gpt == True:
            if "Ministral" in train_config.model_name:
                prompt = [
                    {"role": "user", "content":  f"{system_prompt}\n\nQuestion: {sample['question']}"},
                    {"role": "assistant", "content": f"Response:{sample['response_clean']}"},
                    ]
            else:
                prompt = [{'role': 'system', 'content': system_prompt},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"},
                ]
            return {
                "prompt": prompt,
                'y': sample['y']
            }
        else:
            if "Ministral" in train_config.model_name:
                prompt = [
                    {"role": "user", "content":  f"{system_prompt}\n\nQuestion: {sample['input']}"},
                    {"role": "assistant", "content": f"Response:"},
                    ]
            else:
                prompt = [{'role': 'system', 'content': system_prompt},
                {"role": "user", "content":  f"Question: {sample['input']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
            correct_answer = "yes" if sample['target_scores']["Yes"] == 1 else "no"
            return {
                'question': json.dumps(sample['input']),
                "prompt": json.dumps(prompt),
                "correct_answer": json.dumps(correct_answer),
            }
    def tokenize_add_label(sample):
        prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
        prompt = torch.cat((prompt, torch.tensor([220]))) # manually add white space because the tokenizer will automatically remove the white space ate the end of the sentence
        response = tokenizer.encode(sample['prompt'][2]['content'], add_special_tokens=False)
        response = torch.cat((torch.tensor(response), torch.tensor([220])))
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            'label': [-100] * (len(prompt)-len(response)) + response.tolist(),
            'y': [sample['y']]
            }

        return sample

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    if train_config.train_gpt == True:
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

