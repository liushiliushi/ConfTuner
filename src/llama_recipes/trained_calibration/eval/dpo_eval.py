import argparse 
import json
import numpy as np 
from pathlib import Path

from sklearn.metrics import roc_auc_score
from calibration_metric.metric import ECEMetric

def read_file(path):
    with open(path) as f1:
        data = [json.loads(x) for x in f1.readlines()]

    return data

def main(args):
    data = read_file(args.eval_file)
    data_by_type = {"trained": {"tp":[], "fp":[], "fn":[]}, "reference": {"tp":[], "fp":[], "fn":[]}}

    scores_for_auroc = {"trained": {"probs": [], "corrects": [], "nonanswers": []}, "reference": {"probs": [], "corrects": [], "nonanswers": []}}


    for row in data:
        ref_prob = row['reference_prob']
        trained_prob = row['trained_prob']
        ref_correct = row['reference_correct']
        trained_correct = row['trained_correct']

        # don't include examples where either model gave no answer
        if row['reference_answer'] == "NONE": 
            scores_for_auroc['reference']['nonanswers'].append(True)
        else:
            scores_for_auroc['reference']['nonanswers'].append(False)
        if row['trained_answer'] == "NONE": 
            scores_for_auroc['trained']['nonanswers'].append(True)
        else:
            scores_for_auroc['trained']['nonanswers'].append(False)  
        if args.skip_none:
            if row['reference_answer'] == "NONE" or row['trained_answer'] == "NONE":
                continue

        scores_for_auroc["trained"]["probs"].append(trained_prob)
        scores_for_auroc["trained"]["corrects"].append(trained_correct)
        scores_for_auroc["reference"]["probs"].append(ref_prob)
        scores_for_auroc["reference"]["corrects"].append(ref_correct)

        ref_accept = ref_prob > args.threshold
        trained_accept = trained_prob > args.threshold

        data_by_type["trained"]["tp"].append(trained_accept and trained_correct)
        data_by_type["trained"]["fp"].append(trained_accept and not trained_correct)
        data_by_type["trained"]["fn"].append(not trained_accept and trained_correct)

        data_by_type["reference"]["tp"].append(ref_accept and ref_correct)
        data_by_type["reference"]["fp"].append(ref_accept and not ref_correct)
        data_by_type["reference"]["fn"].append(not ref_accept and ref_correct)

    data_to_write = {}
    for model_type, model_data in data_by_type.items():
        tp = np.sum(model_data["tp"])
        fp = np.sum(model_data["fp"])
        fn = np.sum(model_data["fn"])

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)

        acc = np.mean(scores_for_auroc[model_type]['corrects'])

        print(f"Model Type: {model_type}")
        print(f"Precision: {prec*100:.2f}")
        print(f"Recall: {rec*100:.2f}")
        print(f"F1: {f1*100:.2f}")
        print(f"Accuracy: {acc*100:.2f}")
        abstention_rate = np.mean(scores_for_auroc[model_type]['nonanswers'])
        print(f"Rate of nonanswers: {np.mean(scores_for_auroc[model_type]['nonanswers'])*100:.2f}")

        preds, corrects = scores_for_auroc[model_type]["probs"], scores_for_auroc[model_type]["corrects"]
        auroc = roc_auc_score(corrects, preds)
        print(f"AUROC: {auroc}")

        # get ECE 
        try:
            metric = ECEMetric(n_bins=9)
            ece = metric(np.array(preds), np.array(corrects))
            print(f"ECE: {ece*100:.2f}")
        except IndexError:
            ece = np.nan


        print()
        data_to_write[model_type] = {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc, "auroc": auroc, "ece": ece, "abstenion": abstention_rate}

    # reference accuracy on nonanswers
    trained_nonanswers = scores_for_auroc["trained"]["nonanswers"]
    reference_corrects = scores_for_auroc["reference"]["corrects"]
    corrects_when_answered = [x for x, y in zip(reference_corrects, trained_nonanswers) if not y]
    corrects_when_not_answered = [x for x, y in zip(reference_corrects, trained_nonanswers) if y]
    print(f"reference accuracy on nonanswers when trained model did not answer: {np.mean(corrects_when_not_answered)*100:.2f}")
    print(f"reference accuracy on nonanswers when trained model did answer: {np.mean(corrects_when_answered)*100:.2f}")

    out_path = Path(args.eval_file).parent
    skip_str = "skip" if args.skip_none else "noskip"
    with open(out_path / f"eval_data_{skip_str}.json", "w") as f1:
        json.dump(data_to_write, f1)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--skip_none", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    main(args)