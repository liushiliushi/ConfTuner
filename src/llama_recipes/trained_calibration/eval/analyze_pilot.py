import argparse

import csv
import json 


def read_csv_file(path):
    with open(path) as f1:
        reader = csv.DictReader(f1) 
        return [row for row in reader]


def main(args):
    csv_data = read_csv_file(args.results_file)
    expected = {"accepts": [True, False, True, False],
                "confidence_lower_upper": [(75, 100), (75, 100), (50, 100), (50, 100)],
                "knowledge_lower_upper": [(75, 100), (75, 100), (50, 75), (50, 75)],
                "convincing_lower_upper": [(75, 100), (50, 100), (75, 100), (50, 75)]
                }


    good_anns = []
    for batch in csv_data:
        annotator = batch['WorkerId']
        accepts = json.loads(batch['Answer.choiceList'])
        confidences = json.loads(batch['Answer.sliderValueList'])
        knew_answer = json.loads(batch['Answer.ownSliderValueList'])
        teammate_convincing = json.loads(batch['Answer.teammateSliderValueList'])

        accepts = [x == "accept" for x in accepts]
        all_passed = True
        if accepts == expected['accepts']:
            print(f"Annotator: {annotator} passes accepts")
        else:
            all_passed = False
            print(f"Annotator: {annotator} fails accepts")
            print(accepts)
        conf_in_range = [x >= y[0] and x <= y[1] for x, y in zip(confidences, expected['confidence_lower_upper'])]
        if all(conf_in_range):
            print(f"Annotator: {annotator} passes confidence")
        else:
            all_passed = False
            print(f"Annotator: {annotator} fails confidence")
            print(confidences)
        
        knew_in_range = [x >= y[0] and x <= y[1] for x, y in zip(knew_answer, expected['knowledge_lower_upper'])]
        if all(knew_in_range):
            print(f"Annotator: {annotator} passes knowledge")
        else:
            all_passed = False
            print(f"Annotator: {annotator} fails knowledge")
            print(knew_answer)

        convincing_in_range = [x >= y[0] and x <= y[1] for x, y in zip(teammate_convincing, expected['convincing_lower_upper'])]
        if all(convincing_in_range):
            print(f"Annotator: {annotator} passes convincing")
        else:
            all_passed = False
            print(f"Annotator: {annotator} fails convincing")
            print(teammate_convincing)

        if all_passed:
            good_anns.append(annotator)

    print(f"Good annotators: {good_anns}, len(good_anns): {len(good_anns)}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True, help="path to results csv")
    args = parser.parse_args()
    main(args)