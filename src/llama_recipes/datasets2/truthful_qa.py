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

system_prompt = """You will be asked questions. Please respond to the best of your ability.
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
system_prompt_linguistic = """You will be asked questions. Please respond to the best of your ability.
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

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()

def get_truthful_qa_reflection(tokenizer, split, vllm=True):
    dataset = datasets.load_dataset("truthful_qa", "generation", cache_dir="../dataset/Truthful_qa_raw", split="validation")
    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': system_prompt_reflection},
            {"role": "user", "content":  f"Question: {sample['question']}"},
            {"role": "assistant", "content": f"Response:"},
            ]
        return {
            'question': json.dumps(sample['question']),
            "prompt": json.dumps(prompt),
            "correct_answer": json.dumps(sample['correct_answers']),
        }
    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    return dataset

def get_truthful_qa_raw(tokenizer, split, train_config, vllm=True):
    
    dataset = datasets.load_dataset("truthful_qa", "generation", cache_dir="../dataset/Truthful_qa_raw", split=split)


    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': system_prompt},
                  {"role": "user", "content": f"Question: {sample['question']}"},
                  {"role": "assistant", "content": f"Response:"}
                  ]
        prompt = json.dumps(prompt)
        correct_answers = sample['correct_answers']
        correct_answer = json.dumps(correct_answers)
        return {
            'question': sample['question'],
            "prompt": prompt,
            "correct_answer": correct_answer,
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    return dataset

def get_truthful_qa(tokenizer, split, train_config):
    if split == 'train':
        path = "../dataset/truthful_qa/tqa_train_response.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    elif split == 'val':
        if train_config.train_gpt:
            path = "../dataset/truthful_qa/validation_gpt_temp=0.jsonl"
        else:
            path = "../dataset/truthful_qa/validation_response_temp=0.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train')
    else:
        dataset = datasets.load_dataset("truthful_qa", "generation", cache_dir="../dataset/Truthful_qa_raw", split="validation")

    def apply_prompt_template(sample):
        if "Ministral" in train_config.model_name:
            prompt = [
                {"role": "user", "content":  f"{system_prompt}\n\nQuestion: {sample['question']}"},
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
            "correct_answer": json.dumps(sample['correct_answers']),
        }

    if split == 'test':
        dataset = dataset.map(apply_prompt_template_test, remove_columns=list(dataset.features))
    else:
        dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

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

    if split == 'test':
        dataset = dataset
    else:
        dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    return dataset

