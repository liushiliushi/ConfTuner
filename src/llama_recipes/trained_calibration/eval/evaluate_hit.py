import pdb 
import argparse
import csv
import json 
import numpy as np
from collections import defaultdict


def read_json_file(path):
    with open(path) as f1:
        return [json.loads(x) for x in f1.readlines()]
    
def read_csv_file(path):
    with open(path) as f1:
        reader = csv.DictReader(f1) 
        return [row for row in reader]


def get_annotations_by_prompt(prompt, json_data):
    prompt = prompt.split("[SEP]")[0].strip()
    json_data = [x for x in json_data if x['prompt'] == prompt]
    return json_data


def min_max_normalize(scores):
    return [(x - min(scores)) / (max(scores) - min(scores)) for x in scores]

def normalize_scores_by_ann(batches_by_ann):
    for ann, batches in batches_by_ann.items():
        for i, batch in enumerate(batches):
            for key in ['confidences', 'knew_answer', 'teammate_convincing']:
                batch[f"{key}_normalized"] = min_max_normalize(batch[key])
            batches[i] = batch

        batches_by_ann[ann] = batches
    return batches_by_ann

def check_attention_checks(batches_by_ann):
    # check and remove attention checks 
    for ann, batches in batches_by_ann.items():
        for i, batch in enumerate(batches):
            to_remove = []
            for j, id in enumerate(batch['ids']): 
                was_accept = batch['accepts'][j]
                was_knowledge = batch['knew_answer_normalized'][j]
                was_confident = batch['confidences_normalized'][j]
                if id.startswith("attention_check"):
                    type_key = id.split("[SEP]")[1].strip()
                    accept_type, confidence_type, knowledge_type = type_key.split("_")  
                    if accept_type == "accept":
                        if was_accept == "reject":
                            print(f"accept attention check failed for {ann} {id}")
                    # actually not going to filter based on confidence since it's subjective 
                    # if confidence_type == "confident" and was_confident < 0.5 or confidence_type == "unconfident" and was_confident > 0.5:
                        # print(f"confidence attention check failed for {ann} {id}")

                    if knowledge_type == "knowledge" and was_knowledge < 0.5 or knowledge_type == "noknowledge" and was_knowledge > 0.5:
                        print(f"knowledge attention check failed for {ann} {id}")

                    to_remove.append(j)
            # print(f"removing {len(to_remove)} attention checks for {ann}") 
            for key in ['confidences', 'knew_answer', 'teammate_convincing', 
                        'accepts', 'ids', "answers",
                        "confidences_normalized", "knew_answer_normalized", 
                        "teammate_convincing_normalized"]:
                batch[key] = [x for i, x in enumerate(batch[key]) if i not in to_remove]
            
            batches[i] = batch
        batches_by_ann[ann] = batches
    return batches_by_ann


def get_accept_rate(csv_data, json_data, key="trained"): 
    qual_data = {"true_positives": [], "false_positives": [], "true_negatives": [], "false_negatives": []}
    true_accepts = []
    false_accepts = []
    true_rejects = []
    false_rejects = []
    all_p_accept = []
    skipped = 0
    batches_by_ann = defaultdict(list)
    for batch in csv_data:
        accepts = json.loads(batch['Answer.choiceList'])
        confidences = json.loads(batch['Answer.sliderValueList'])
        knew_answer = json.loads(batch['Answer.ownSliderValueList'])
        teammate_convincing = json.loads(batch['Answer.teammateSliderValueList'])
        answers = json.loads(batch['Input.inputAnswerList']) 


        ids = json.loads(batch['Input.inputQuestionIdList'])
        ann = batch['WorkerId']

        row_dict = {"accepts": accepts,
                    "confidences": confidences,
                    "knew_answer": knew_answer,
                    "teammate_convincing": teammate_convincing,
                    "ids": ids,
                    "answers": answers}
        batches_by_ann[ann].append(row_dict)

    batches_by_ann = normalize_scores_by_ann(batches_by_ann)
    batches_by_ann = check_attention_checks(batches_by_ann)

    total = 0
    for ann, data in batches_by_ann.items():
        for batch in data:
            accepts = batch['accepts']
            confidences = batch['confidences_normalized']
            knew_answer = batch['knew_answer_normalized']
            teammate_convincing = batch['teammate_convincing_normalized']
            ids = batch['ids']
            answers = batch['answers']

            prompts, is_traineds = zip(*[x.split("[SEP]") for x in ids])


            for prompt, is_trained, accept, conf, knew, convincing, answer in zip(prompts, is_traineds, accepts, confidences, knew_answer, teammate_convincing, answers):
                if is_trained == key:
                    # they knew
                    if knew > 0.5:
                        skipped += 1
                        continue

                    total += 1
                
                    anns = get_annotations_by_prompt(prompt, json_data)[0]
                    is_correct = anns[f"{key}_correct"]     


                    if accept == "reject":
                        # first flip it, since high conf of reject is low conf of accept 
                        conf = 1-conf
                        # p_accept is lower bounded at 0 and upper bounded at 0.5
                        # normalize conf so that 0 is 0 and 1 is 0.5
                        conf = conf * 0.5
                        p_accept = conf
                    else:
                        # p_accept is lower bounded at 0.5 and upper bounded at 1
                        # normalize conf so that 0 is 0.5 and 1 is 1
                        conf = 0.5 + conf * 0.5 
                        p_accept = conf
                    all_p_accept.append(p_accept)



                    if is_correct and accept == "accept":
                        true_accepts.append(p_accept)
                        qual_data['true_positives'].append(f"Question: {prompt}\nAnswer: {answer}\n p_accept {p_accept}, knowledge: {knew}, convincing: {convincing}, ann: {ann}")
                    elif not is_correct and accept == "accept":
                        false_accepts.append(p_accept)
                        qual_data['false_positives'].append(f"Question: {prompt}\nAnswer: {answer}\n p_accept {p_accept}, knowledge: {knew}, convincing: {convincing} ann: {ann}") 
                    elif is_correct and accept == "reject":
                        true_rejects.append(p_accept)
                        qual_data['true_negatives'].append(f"Question: {prompt}\nAnswer: {answer}\n p_accept {p_accept}, knowledge: {knew}, convincing: {convincing}, ann: {ann}") 
                    elif not is_correct and accept == "reject":
                        false_rejects.append(p_accept)
                        qual_data['false_negatives'].append(f"Question: {prompt}\nAnswer: {answer}\n p_accept {p_accept}, knowledge: {knew}, convincing: {convincing}, ann: {ann}") 
                    else:
                        raise ValueError("unknown value")

    print(f"skipped: {skipped} of {total} total")
    print(f"average p accept: {sum(all_p_accept) / len(all_p_accept)}")
    print(f"{key} true accepts: {len(true_accepts)} with p_accept {np.mean(true_accepts)}")
    print(f"{key} false accepts: {len(false_accepts)} with p_accept {np.mean(false_accepts)}")
    print(f"{key} true rejects: {len(true_rejects)} with p_accept {np.mean(true_rejects)}")
    print(f"{key} false rejects: {len(false_rejects)}, with p_accept {np.mean(false_rejects)}")

    precision = len(true_accepts) / (len(true_accepts) + len(false_accepts))
    print(f"{key} precision: {precision}")
    print()

    for inner_key, values in qual_data.items():
        with open(f"analysis/hit_results/qualitative/{key}_{inner_key}_qualitative.txt", "w") as f1:
            for value in values:
                f1.write(f"{value}\n")
                f1.write("\n\n")

def main(args):
    json_data = read_json_file(args.json_file)
    csv_data = read_csv_file(args.results_file)

    # get accept rate 
    get_accept_rate(csv_data, json_data, key="trained") 
    get_accept_rate(csv_data, json_data, key="reference")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str, required=True, help="path to eval_dpo_on_valid.jsonl file")
    parser.add_argument("--results_file", type=str, required=True, help="path to results csv")
    parser.add_argument("--out_dir", type=str, required=False, default="analysis/hit_results/qualitative/")
    args = parser.parse_args()

    main(args)