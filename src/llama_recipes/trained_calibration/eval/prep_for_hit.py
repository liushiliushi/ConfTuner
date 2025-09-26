import argparse 
import json 
import csv 
import numpy as np
import pdb 
import re 
np.random.seed(12)

from trained_calibration.eval.attention_checks import attention_check_data

def read_file(path):
    with open(path) as f1:
        return [json.loads(x) for x in f1.readlines()]

def resubstitute(response, answer):
    return re.sub("\[ANSWER REMOVED\]", answer, response)

def get_question(prompt): 

    return re.search("Question: (.*)Response", prompt, flags=re.DOTALL).group(1).strip()

def get_batch_row(batches): 

    to_ret = []
    for batch in batches:
        # for each batch, alternate between trained and reference 
        csv_batch_1 = {"inputQuestionList": [], "inputAnswerList": [], "inputQuestionIdList": []}
        csv_batch_2 = {"inputQuestionList": [], "inputAnswerList": [], "inputQuestionIdList": []}

        # add attention checks to each batch 
        attention_checks_b1 = np.random.choice(attention_check_data, 2, replace=False)
        attention_checks_b2 = np.random.choice(attention_check_data, 2, replace=False)
        added_b1, added_b2 = 0, 0
        for i, row in enumerate(batch):
            # if even, batch1 gets trained, batch2 gets reference
            question = get_question(row['prompt'])

            p_add_b1 = np.random.uniform()
            p_add_b2 = np.random.uniform()
            # if we haven't added both, add one  
            if added_b1 < 2 and p_add_b1 > 0.75:
                attention_check = attention_checks_b1[added_b1]
                csv_batch_1["inputQuestionList"].append(attention_check['question'])
                csv_batch_1["inputAnswerList"].append(attention_check['answer'])
                csv_batch_1["inputQuestionIdList"].append(attention_check['id'])
                added_b1 += 1
            if added_b2 < 2 and p_add_b2 > 0.75:
                attention_check = attention_checks_b2[added_b2]
                csv_batch_2["inputQuestionList"].append(attention_check['question'])
                csv_batch_2["inputAnswerList"].append(attention_check['answer'])
                csv_batch_2["inputQuestionIdList"].append(attention_check['id'])
                added_b2 += 1

            if i % 2 == 0:
                csv_batch_1["inputQuestionList"].append(question)
                csv_batch_1["inputAnswerList"].append(resubstitute(row['trained_output'], row['trained_answer']))
                csv_batch_1["inputQuestionIdList"].append(f"{row['prompt']}[SEP]trained") 

                csv_batch_2["inputQuestionList"].append(question)
                csv_batch_2["inputAnswerList"].append(resubstitute(row['reference_output'], row['reference_answer']))
                csv_batch_2["inputQuestionIdList"].append(f"{row['prompt']}[SEP]reference") 
            else:
                csv_batch_1["inputQuestionList"].append(question) 
                csv_batch_1["inputAnswerList"].append(resubstitute(row['reference_output'], row['reference_answer']))
                csv_batch_1["inputQuestionIdList"].append(f"{row['prompt']}[SEP]reference")

                csv_batch_2["inputQuestionList"].append(question) 
                csv_batch_2["inputAnswerList"].append(resubstitute(row['trained_output'], row['trained_answer']))
                csv_batch_2["inputQuestionIdList"].append(f"{row['prompt']}[SEP]trained")

        # add the last ones in if they haven't been done yet  
        if added_b1 < 2:
            # in case both weren't added
            for i in range(2-added_b1):
                attention_check = attention_checks_b1[i]
                csv_batch_1["inputQuestionList"].append(attention_check['question'])
                csv_batch_1["inputAnswerList"].append(attention_check['answer'])
                csv_batch_1["inputQuestionIdList"].append(attention_check['id'])
        if added_b2 < 2: 
            for i in range(2-added_b2):
                attention_check = attention_checks_b2[i]
                csv_batch_2["inputQuestionList"].append(attention_check['question'])
                csv_batch_2["inputAnswerList"].append(attention_check['answer'])
                csv_batch_2["inputQuestionIdList"].append(attention_check['id'])

        to_ret.append({k:json.dumps(v) for k, v in csv_batch_1.items()})
        to_ret.append({k:json.dumps(v) for k, v in csv_batch_2.items()})


    return to_ret

def get_stratified_samples(data, k, bin_num = 5):
    # stratify by confidence score 
    # stratify by trained prob
    scores = [x['trained_prob'] for x in data] 
    # bin scores like for histogram
    bins = np.histogram(scores, bins=bin_num)[1]
    num_per_bin = k/bin_num
    stratified_data = []

    for i in range(bin_num):
        bin_data = [x for x in data if bins[i] <= x['trained_prob'] < bins[i+1]]
        if len(bin_data) < num_per_bin:
            print(f"bin {i} has fewer than {num_per_bin} samples")
            stratified_data.extend(bin_data)
        else:
            stratified_data.extend(np.random.choice(bin_data, int(num_per_bin), replace=False))

    # if we're under, then add some randomly sampled examples
    existing_prompts = [x['prompt'] for x in stratified_data]
    if len(stratified_data) < k:
        print(f"missing {k - len(stratified_data)} of {num_per_bin} samples, adding random")
    while len(stratified_data) < k: 
        sample_idx = np.random.choice(len(data), 1, replace=False)[0]
        sample = data[sample_idx]
        if sample['prompt'] in existing_prompts:
            continue
        stratified_data.append(sample)
        existing_prompts.append(sample['prompt'])

    pdb.set_trace()
    return stratified_data


 
def main(args):
    data = read_file(args.eval_file)

    # for human eval, we will sample 100 questions 
    # 50 that the model got right
    # 50 that the model got wrong 
    # exclude any examples where either model is NONE

    np.random.shuffle(data)
    print(len(data))
    correct_data = [x for x in data if x['trained_correct']]
    incorrect_data = [x for x in data if not x['trained_correct']]
    print(len(correct_data))
    print(len(incorrect_data))
    for_hit_correct, for_hit_incorrect = [], []
    half_limit = args.limit//2

    if args.stratify:
        for_hit_correct.extend(get_stratified_samples(correct_data, half_limit))
        for_hit_incorrect.extend(get_stratified_samples(incorrect_data, half_limit))
    else:
        for row in correct_data:
            if len(for_hit_correct) == half_limit:
                break
            if row['trained_answer'] == "NONE" or row['reference_answer'] == "NONE":
                continue
            for_hit_correct.append(row)
        for row in incorrect_data:
            if len(for_hit_incorrect) == half_limit:
                break
            if row['trained_answer'] == "NONE" or row['reference_answer'] == "NONE":
                continue
            for_hit_incorrect.append(row)
    for_hit_data = for_hit_correct + for_hit_incorrect
    print(len(for_hit_data)) 
    np.random.shuffle(for_hit_data)


    # break into batches of 20 
    batches = []
    for i in range(0, len(for_hit_data), 20):
        batch = for_hit_data[i:i+20]
        batches.append(batch)

    output_data = get_batch_row(batches)
    pdb.set_trace()
    with open(args.out_file, "w") as f1:
        writer = csv.DictWriter(f1, fieldnames=["inputQuestionList", "inputAnswerList", "inputQuestionIdList"])
        writer.writeheader()
        for row in output_data:
            writer.writerow(row)


    # for each question, we have the reference question and the model question
    # so 200 examples total, with 20 questions per HIT 
    # no redundancy 

    # questions
        # does it matter if the same annotator sees the reference and original? I think it's fine as long as they aren't in the same assignment 


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True, help="path to eval_dpo_on_valid.jsonl file")
    parser.add_argument("--out_file", type=str, required=True, help="path to output file")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--stratify", action="store_true", help="set to true if you want to stratify by confidence")
    args = parser.parse_args()

    main(args)