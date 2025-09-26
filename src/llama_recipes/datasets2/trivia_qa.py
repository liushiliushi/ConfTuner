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

system_prompt = """You will be asked trivia questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Then please extract a single answer from the your response. If no answer is present, please write "NONE".
            Finally, please provide your confidence (0%-100%) to your answer.

            Here are some examples:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was John Milton, who published the book in 1667.
            Final answer: John Milton
            Confidence: 90%

            Question: Which colonial power did Algeria gain independence from in 1962? 
            Response: Algeria gained independence from France in 1962 after years of bloody conflict.
            Final answer: France
            Confidence: 100%

            Question: How many planets are in our solar system?
            Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
            Final answer: NONE
            Confidence: 0%"""


system_prompt_linguistic = """You will be asked trivia questions. Please respond to the best of your ability.
            Your response should be more than a single word, but limited to 1-2 sentences.
            Then please extract a single answer from the your response. If no answer is present, please write "NONE".
            Assess your confidence level based on:
                    - High (66%-100%): Certain of correctness with logical reasoning
                    - Medium (33%-66%): Partially confident but some uncertainty
                    - Low (0%-33%): Suspect potential errors in calculation/logic

                    
            Here are some examples:

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was John Milton, who published the book in 1667.
            Final answer: John Milton
            Confidence: high

            Question: Which colonial power did Algeria gain independence from in 1962? 
            Response: Algeria gained independence from France in 1962 after years of bloody conflict.
            Final answer: France
            Confidence: high

            Question: How many planets are in our solar system?
            Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
            Final answer: NONE
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



def get_trivia_qa_raw(tokenizer, split, train_config, vllm=True):
    
    if split == 'train':
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.web.nocontext", cache_dir="../dataset/Trivia_qa_raw", split='train[:10000]')
    else:
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", "rc.web.nocontext", cache_dir="../dataset/Trivia_qa_raw", split='validation[:1000]')

    def apply_prompt_template(sample):
        prompt = [{'role': 'system', 'content': system_prompt},
                  {"role": "user", "content": f"Question: {sample['question']}"},
                  {"role": "assistant", "content": f"Response:"}
                  ]
        prompt = json.dumps(prompt)
        correct_answers = sample['answer']['normalized_aliases'] + [normalize_answer(ans) for ans in sample['answer'].get('human_answers', [])]
        correct_answer = json.dumps(correct_answers)
        return {
            'question': sample['question'],
            "prompt": prompt,
            "correct_answer": correct_answer,
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset

def get_trivia_qa(tokenizer, split, train_config):
    if split == 'train':
        path = "../dataset/trivia_qa/train_response_temp=0.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[:2000]')
    elif split == 'val':
        if train_config.train_gpt:
            path = "../dataset/trivia_qa/validation_gpt_temp=0.jsonl"
        else:
            path = "../dataset/trivia_qa/validation_response_temp=0.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')
    else:
        path = "../dataset/trivia_qa/validation_response_temp=0.jsonl"
        dataset = datasets.load_dataset('json', data_files=path, split='train[:1000]')

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
        matches1 = re.findall("Final answer: (.*)", sample['response_clean'])
        matches2 = re.findall("Confidence:", sample['response_clean'])
        if matches1 and matches2:
            answer = re.findall("Final answer: (.*)", sample['response_clean'])[-1]
            y  = 1 if normalize_answer(answer).lower().strip() in sample['correct_answer'] else 0
            return {
                "prompt": prompt,
                "y": y,
            }
        else:
            print("Error")
            print(sample['response_clean'])
            return {
                "prompt": prompt,
                "y": 0,
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

