import copy
import datasets
import os
import json
import os
import re
import torch
from datasets import concatenate_datasets
from functools import partial
import string
def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

system_prompt = """You will be asked reasoning questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Finally, please provide your confidence (0%-100%) to your answer.

            Here are some examples:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was John Milton, who published the book in 1667.
            Confidence: 90%

            Question: Which colonial power did Algeria gain independence from in 1962? 
            Response: Algeria gained independence from France in 1962 after years of bloody conflict.
            Confidence: 100%

            Question: How many planets are in our solar system?
            Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
            Confidence: 0%"""

system_prompt_reflection = """For the question, response, and confidence, if the confidence is less than 50%, please revise your response and provide a better one. Otherwise, please repeat the response and the confidence.

            Here is the example:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was Percy Bysshe Shelley.
            Confidence: 40%
            If the confidence is less than 50%, analyze the answer and provide a better one. 
            Reflection: The response is less than 50%. 
            Response: The author of Paradise Lost wasn't Percy Bysshe Shelley, it was John Milton, who published the book in 1667.
            Confidence: 90%
            
            """
system_prompt_linguistic = """You will be asked reasoning questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Assess your confidence level based on:
                    - High (66%-100%): Certain of correctness with logical reasoning
                    - Medium (33%-66%): Partially confident but some uncertainty
                    - Low (0%-33%): Suspect potential errors in calculation/logic

            Here are some examples:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was John Milton, who published the book in 1667.
            Confidence: high

            Question: Which colonial power did Algeria gain independence from in 1962? 
            Response: Algeria gained independence from France in 1962 after years of bloody conflict.
            Confidence: high

            Question: How many planets are in our solar system?
            Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
            Confidence: low"""


system_prompt_coarse = """You will be asked reasoning questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Finally, please provide your confidence (0-9) to your answer. 
            The confidence score must be a value between 0-9, where 9 is the maximum. Never use 10.

            Here are some examples:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was John Milton, who published the book in 1667.
            Confidence: 8

            Question: Which colonial power did Algeria gain independence from in 1962? 
            Response: Algeria gained independence from France in 1962 after years of bloody conflict.
            Confidence: 9

            Question: How many planets are in our solar system?
            Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
            Confidence: 0"""

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_hotpot_qa_reflection(tokenizer, split, vllm=True):
    path = "../dataset/hotpot_qa/validation_response_temp=0.jsonl"
    dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')
    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': system_prompt_reflection},
                  {"role": "user", "content": f"Question: {sample['question']}"},
                  {"role": "assistant", "content": f"Response:"}
                  ]

        return {
            'question': json.dumps(sample['question']),
            "prompt": json.dumps(prompt),
            "correct_answer": json.dumps(sample['correct_answer']),
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

    

def get_hotpot_qa_raw(tokenizer, split, train_config, vllm=True):
    
    if split == 'train':
        dataset = datasets.load_dataset("hotpotqa/hotpot_qa",'distractor', cache_dir="../dataset/Hotpot_qa_raw", split='train[:12000]', trust_remote_code=True)
    elif split == "validation":
        dataset = datasets.load_dataset("hotpotqa/hotpot_qa", 'distractor', cache_dir="../dataset/Hotpot_qa_raw", split='validation[:2000]', trust_remote_code=True)
    def apply_prompt_template(sample):
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content":  f"{system_prompt}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"}
            ]
        else:
            prompt = [{'role': 'system', 'content': system_prompt},
                  {"role": "user", "content": f"Question: {sample['question']}"},
                  {"role": "assistant", "content": f"Response:"}
                  ]
        prompt = json.dumps(prompt)
        correct_answer = json.dumps(sample['answer'])
        return {
            'question': sample['question'],
            "prompt": prompt,
            "correct_answer": correct_answer,
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

def get_hotpot_qa(tokenizer, split, train_config):
    if split == 'train':
        if "Ministral" in train_config.model_name:
            path = "../dataset/hotpot_qa/train_ministral_temp=0.jsonl"
        elif "Llama-3.1" in train_config.model_name:
            if train_config.train_gpt:
                path = "../dataset/hotpot_qa/train_response_gpt.jsonl"
            else:
                path = "../dataset/hotpot_qa/train_llama_temp=0.jsonl"
        elif "Qwen" in train_config.model_name:
            if train_config.train_gpt:
                path = "../dataset/hotpot_qa/train_response_gpt.jsonl"
            else:
                path = "../dataset/hotpot_qa/train_Qwen_temp=0.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[:2000]')
    elif split == 'val':
        if train_config.train_gpt:
            path = "../dataset/hotpot_qa/validation_response_gpt.jsonl"
        else:
            path = "../dataset/hotpot_qa/validation_response_temp=0.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')
    else:
        path = "../dataset/hotpot_qa/validation_response_temp=0.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')

    def apply_prompt_template(sample):
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content":  f"{system_prompt_coarse}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
            ]
        elif "Qwen" in train_config.model_name:
            prompt = [{'role': 'system', 'content': system_prompt_coarse},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
                ]
        else:
            prompt = [{'role': 'system', 'content': system_prompt},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:{sample['response_clean']}"}
                ]
        return {
            "prompt": prompt,
            "y": sample['y'],
        }
        
        
    def apply_prompt_template_test(sample):
        global system_prompt
        if train_config.test_linguistic:
            system_prompt = system_prompt_linguistic
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content": f"{system_prompt}\n\nQuestion: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        else:
            prompt = [{'role': 'system', 'content': system_prompt},
                {"role": "user", "content":  f"Question: {sample['question']}"},
                {"role": "assistant", "content": f"Response:"},
                ]
        return {
            'question': json.dumps(sample['question']),
            "prompt": json.dumps(prompt),
            "correct_answer": json.dumps(sample['correct_answer']),
        }

    if split == 'test':
        dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features))
    else:
        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.apply_chat_template(sample['prompt'], tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).squeeze(0)
        prompt = torch.cat((prompt, torch.tensor([220]))) # manually add white space because the tokenizer will automatically remove the white space ate the end of the sentence

        if "Ministral" in train_config.model_name:
            response = torch.tensor(tokenizer.encode(sample['prompt'][1]['content'], add_special_tokens=False))
        else:
            response = torch.tensor(tokenizer.encode(sample['prompt'][2]['content'], add_special_tokens=False))
        sample = {
            "input_ids": prompt,
            "attention_mask" : [1] * (len(prompt)),
            'label': [-100] * (len(prompt)-len(response)) + response.tolist(),
            'y': [sample['y']]
            }

        return sample

    if split == 'test':
        dataset = dataset
    else:
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset
