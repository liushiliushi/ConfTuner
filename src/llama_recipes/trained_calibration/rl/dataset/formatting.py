import json
import re 
import datasets 
import pandas as pd 
import numpy as np
import pdb 
from collections import defaultdict

def just_response(response):
    response = re.split("Response:", response)[1]
    return response


def sub_answer(response, answer):
    # sub answer back in 
    if len(response.strip()) == 0:
        return answer

    response = re.sub("\[ANSWER REMOVED\]", answer, response)
    return response

def compute_baseline_rank(response1, response2):
    """
    compute preference rank based only on correctness
    """
    r1c, r2c = response1["correct"], response2["correct"]
    # if both the same, can't compare
    if r1c == r2c:
        return None
    elif r1c and not r2c:
        return 0
    return 1

def compute_rank(response1, response2, thresh, min_prob_margin, mapping_fxn=None):
    """
    compute preference rank of two examples according to both probability of acceptance and whether it was correct
    """
    # ranking order: 
    # c,a > ~c,~a > c,~a > ~c,a  
    r1c, r2c = response1["correct"], response2["correct"]
    r1p, r2p = response1["mean_prob"], response2["mean_prob"]
    r1a, r2a = r1p > thresh, r2p > thresh

    # if the difference in probabilities is very small, don't rank, not meaningful
    if abs(r1p - r2p) < min_prob_margin:
        return None, None

    if mapping_fxn is None:
        # correct, accept
        mapping_fxn = {(True,True): 1.0, # correctly accepting is worth full reward
                    (False, False): 1.0, # correct rejection is also worth full reward
                    (True, False): -0.5, # false rejection is bad
                    (False, True): -1.0} # false acceptance is worse
    

    r1_score = mapping_fxn[(r1c, r1a)]
    r2_score = mapping_fxn[(r2c, r2a)]

    pref_lut = {(True, True): "c,a",
                (False, False): "~c,~a",
                (True, False): "c,~a",
                (False, True): "~c,a"}
    pref_type_left = pref_lut[(r1c, r1a)]
    pref_type_right = pref_lut[(r2c, r2a)]


    # edge case: with the current scoring, reject + false is better than reject + true
    # I think that is ok 
    # don't rank things that are equally good or bad 
    if r1_score == r2_score:
        return None, None
    if r1_score > r2_score:
        pref_type_str = f"{pref_type_left} > {pref_type_right}"
        return 0, pref_type_str 

    pref_type_str = f"{pref_type_right} > {pref_type_left}"
    return 1, pref_type_str

def format_dataset(path, filter_short = True, min_prob_margin=0.1, limit_per_q = 3, balance_types=False, balance_fp_tp=False, mapping_fxn=None):
    # read data 
    with open(path) as f1:
        data = [json.loads(line) for line in f1]

    # get the threshold, the 50th percentile  
    # to account for the "yes" bias
    probs = [x['mean_prob'] for x in data]
    thresh = np.percentile(probs, 50) 

    # group by idx 
    data_by_prompt = defaultdict(list)
    for d in data:
        if filter_short:
            response = d['response_clean']
            check_response = re.sub("\[ANSWER REMOVED\]", "", response)
            # skip anything which does not have any data besides the answer 
            if len(check_response.strip()) == 0:
                continue

        # skip invalid answers 
        if d['answer'] == "NONE":
            continue

        data_by_prompt[d['prompt']].append(d)

    # iterate over pairs and get rankings 
    all_data = []
    for prompt, data in data_by_prompt.items():
        done = []
        added_for_q = 0
        for i, response1 in enumerate(data):
            for j, response2 in enumerate(data):
                if i == j:
                    continue 
                if (i, j) in done:
                    continue 
                done.append((i, j))
                done.append((j, i))


                best_idx, preference_type = compute_rank(response1, response2, thresh, min_prob_margin, mapping_fxn=mapping_fxn)
                if best_idx is None:
                    continue
                raw_responses = [response1, response2]
                responses = [response1['response_clean'], response2['response_clean']]
                probs = [response1['mean_prob'], response2['mean_prob']]
                corrects = [response1['correct'], response2['correct']]
                answers = [response1['answer'], response2['answer']]

                try:
                    responses = [just_response(r) for r in responses]
                except IndexError:
                    # very rare edge case where the [ANSWER REMOVED] is subbed into "Response"
                    # skip 
                    continue
                responses = [sub_answer(r, raw_responses[i]['answer']) for i, r in enumerate(responses)]

                preferred = responses[best_idx].strip()
                dispreferred = responses[1 - best_idx].strip()
                prompt = response1['prompt'] + " "
                chosen_prob = probs[best_idx]
                rejected_prob = probs[1 - best_idx]
                chosen_correct = corrects[best_idx]
                rejected_correct = corrects[1 - best_idx]
                chosen_answer = answers[best_idx]
                rejected_answer = answers[1 - best_idx]

                # prompt = re.sub("Response:", "Answer:", prompt)
                all_data.append({"prompt": prompt, 
                                 "chosen": preferred, 
                                 "rejected": dispreferred, 
                                 "type": preference_type,
                                 "correct_answers": response1['correct_answers'],
                                 "chosen_prob": chosen_prob,
                                 "rejected_prob": rejected_prob,
                                 "chosen_correct": chosen_correct,
                                 "rejected_correct": rejected_correct,
                                 "chosen_answer": chosen_answer,
                                 "rejected_answer": rejected_answer,})
                added_for_q += 1
                # limit the number we add per question to increase diversity 
                if limit_per_q is not None and added_for_q >= limit_per_q:
                    break

    
    # split into datasets based on type
    type_data = defaultdict(list)
    for d in all_data:
        type_data[d['type']].append(d)
    # always give a report  
    print("Dataset split by type:")
    for t, data in type_data.items():
        print(f"{t}: {len(data)}")

    # if we want balanced types 
    if balance_types:
        min_type_len = min([len(v) for v in type_data.values()])
        print(f"Balancing datasets to have at most {min_type_len} per type.") 
        out_datasets = []
        for t, data in type_data.items():
             
            subdata_idxs = np.random.choice(len(data), min_type_len, replace=False)
            subdata = [data[i] for i in subdata_idxs]
            out_datasets.extend(subdata)
        np.random.shuffle(out_datasets)
        previous_len = len(all_data)
        all_data = out_datasets
        new_len = len(all_data)
        print(f"New balanced dataset has {new_len} examples, down from {previous_len} examples.")

    if balance_fp_tp:
        raise NotImplementedError
    # tdoo (Elias): just balance some
        tp_num = len(type_data['c,a '])
        tn_num = len(type_data['~c,~a'])
        min_type_len = min([tp_num, tn_num])
        print(f"Balancing tp and tn to have at most {min_type_len} per type.") 
        out_datasets = []
        for t, data in type_data.items():
             
            subdata_idxs = np.random.choice(len(data), min_type_len, replace=False)
            subdata = [data[i] for i in subdata_idxs]
            out_datasets.extend(subdata)
        np.random.shuffle(out_datasets)
        previous_len = len(all_data)
        all_data = out_datasets
        new_len = len(all_data)
        print(f"New balanced dataset has {new_len} examples, down from {previous_len} examples.")



    # convert to an HF dataset
    out_dataset = datasets.Dataset.from_pandas(pd.DataFrame(all_data))

    return out_dataset, thresh

def format_baseline_dataset(path, filter_short = True, limit_per_q = 3):
    # read data 
    with open(path) as f1:
        data = [json.loads(line) for line in f1]


    # group by idx 
    data_by_prompt = defaultdict(list)
    for d in data:
        if filter_short:
            response = d['response_clean']
            check_response = re.sub("\[ANSWER REMOVED\]", "", response)
            # skip anything which does not have any data besides the answer 
            if len(check_response.strip()) == 0:
                continue

        # skip invalid answers 
        if d['answer'] == "NONE":
            continue

        data_by_prompt[d['prompt']].append(d)

    # iterate over pairs and get rankings 
    all_data = []
    for prompt, data in data_by_prompt.items():
        done = []
        added_for_q = 0
        for i, response1 in enumerate(data):
            for j, response2 in enumerate(data):
                if i == j:
                    continue 
                if (i, j) in done:
                    continue 
                done.append((i, j))
                done.append((j, i))


                best_idx = compute_baseline_rank(response1, response2) 
                if best_idx is None:
                    continue

                raw_responses = [response1, response2]
                responses = [response1['response_clean'], response2['response_clean']]
                corrects = [response1['correct'], response2['correct']]
                answers = [response1['answer'], response2['answer']]

                try:
                    responses = [just_response(r) for r in responses]
                except IndexError:
                    # very rare edge case where the [ANSWER REMOVED] is subbed into "Response"
                    # skip 
                    continue
                responses = [sub_answer(r, raw_responses[i]['answer']) for i, r in enumerate(responses)]

                preferred = responses[best_idx].strip()
                dispreferred = responses[1 - best_idx].strip()
                prompt = response1['prompt'] + " "
                chosen_correct = corrects[best_idx]
                rejected_correct = corrects[1 - best_idx]
                chosen_answer = answers[best_idx]
                rejected_answer = answers[1 - best_idx]

                # prompt = re.sub("Response:", "Answer:", prompt)
                all_data.append({"prompt": prompt, 
                                 "chosen": preferred, 
                                 "rejected": dispreferred, 
                                 "correct_answers": response1['correct_answers'],
                                 "chosen_correct": chosen_correct,
                                 "rejected_correct": rejected_correct,
                                 "chosen_answer": chosen_answer,
                                 "rejected_answer": rejected_answer,})
                added_for_q += 1
                # limit the number we add per question to increase diversity 
                if limit_per_q is not None and added_for_q >= limit_per_q:
                    break

    # convert to an HF dataset
    out_dataset = datasets.Dataset.from_pandas(pd.DataFrame(all_data))
    return out_dataset, -1
