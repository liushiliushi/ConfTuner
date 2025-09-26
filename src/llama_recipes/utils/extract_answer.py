import re, pdb
from typing import Tuple, Optional, Dict


def extract_hint_response_top_k(text, K, options: Dict[str, str], task_type, error_log_file):
    def default_postprocess_match(match) -> Tuple[str, str]:
        assert match is not None
        option_key, conf_scale = match.group(1), match.group(2)
        return option_key, conf_scale
    
    def postprocess_match_without_option(match) -> Tuple[str, str]:
        assert match is not None
        answer_option_value = match.group(1).strip()
        
        conf_scale = match.group(2)
        answer_option_key = None
        for option_key, option_value in options.items():
            option_value = option_value.strip().strip(".").lower()
            answer_option_value = answer_option_value.strip().strip(".").lower()
            if answer_option_value in option_value or \
                option_value in answer_option_value:
                answer_option_key = option_key
        
        # assert answer_option_key is not None, (match.group(0), answer_option_value, options)
        if answer_option_key is None:
            print(match.group(0), answer_option_value, options)
            # it returns an answer that does not belong to any of the option values
            return "Z", conf_scale
        
        return answer_option_key, conf_scale
    
    def postprocess_match_open_number(match) -> Tuple[str, str]:
        assert match is not None
        numerical_answer, conf_scale = match.group(1), match.group(2)
        numerical_answer = numerical_answer.replace(",", "") # 1,000 -> 1000
        
        return numerical_answer, conf_scale
        
    
    def process_pipeline(ith):
        # Define five different regular expression patterns
        patterns_multi_choice = [
            rf"(?:G{ith}|Guess {ith}):\s*[\(\[]?([A-Z])[\)\]]?\s*(?:P{ith}|Probability {ith}):\s*(\d+)%*",
            rf"(?:G{ith}|Guess {ith}):\s*[\(\[]?([A-Z])[\)\]]?\s*(?:P{ith}|Probability {ith}):\s*(\d+)%?",
            rf"(?:G{ith}|Guess {ith}):\s*[\(\[]?([A-Z])[\)\]]?\s*(?:P{ith}|Probability {ith}):\s*[\(\[]?(\d+)%?[\(\[]?",
            rf"(?:G{ith}|Guess {ith}):\s*[\(\[]?([A-Z])[\)\]]?\s*.*\s*(?:P{ith}|Probability {ith}):\s*[\(\[]?(\d+)%?[\(\[]?",
        ]
        # sometimes the LLM will directly output the answer rather than the associated option key
        patterns_multi_choice_without_option = [
            rf"(?:G{ith}|Guess {ith}):\s*(.*?)\s*(?:P{ith}|Probability {ith}):\s*(\d+)%*"
        ]
        
        # [\(\[]?([A-Z])[\)\]]?  -> [\(\[]? matches optional ([
        # [\)\]]? matches optional )]
        # most appears in vicuna
        # Note: .* can match any character (except for a newline character) zero or more times
        patterns_multi_choice_werid = [
            r"Answer: [\(\[]?([A-Z])[\)\]]?[,.]?\s+Confidence level: (\d+%)",
            r"Answer: [\(\[]?([A-Z])[\)\]]?[,.]?.*\s+Confidence(?: level)?: (\d+%)",
            r"Answer:\s*[\(\[]?([A-Z])[\)\]]?[,.]?\s+Confidence level:\s*(\d+%)"
        ]
        
        patterns_and_postprocess_multi_choice = []
        patterns_and_postprocess_multi_choice.extend([(pat, default_postprocess_match) for pat in patterns_multi_choice])
        patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_without_option])
        # patterns_and_postprocess_multi_choice.extend([(pat, postprocess_match_without_option) for pat in patterns_multi_choice_werid])
        
        # Define five different regular expression patterns
        patterns_open_number = [
            rf"G{ith}:\s*.*?([0-9,.]+)\s*P{ith}:\s*(\d+)%*",
            rf"G{ith}:\s*.*?([0-9,.]+)\s*.*\s+P{ith}:\s*(\d+)%*",
            
        ]
        
        
        patterns_and_postprocess_open_number = [(pat, postprocess_match_open_number) for pat in patterns_open_number]
        
        
        if task_type == "multi_choice_qa":
            patterns_and_postprocess = patterns_and_postprocess_multi_choice
        elif task_type == "open_number_qa":
            patterns_and_postprocess = patterns_and_postprocess_open_number
        else:
            raise ValueError(f"task_type {task_type} is not supported")
        
        return patterns_and_postprocess
       
    # pre-process
    text = text.replace("(1-100)", "(0-100)")
    text = text.replace("\n", " ")
         
    answers, confs = {}, {}
    match_error = []
    for ith in range(0, K):
        # Try each regular expression pattern in turn, until a match is found or all patterns have been tried
        is_match = False
        patterns_and_postprocess = process_pipeline(ith+1)
        for pattern, match_processor in patterns_and_postprocess:
            match = re.search(pattern, text)
            if not match:
                continue
            else:
                answer, conf = match_processor(match)
                answer, conf = check_validity_answer_conf(answer, conf, task_type, error_log_file)
            if answer is not None and conf is not None:
                is_match = True
                break   
            
                    
        if not is_match:
            match_error.append(ith)
            answer = None
            conf = None
            

        answers[ith] = answer
        confs[ith] = conf

    if answers[0] is None or confs[0] is None:
        # If no match is found, print a message
        print(f"\n\nTop-1 ERROR: {match_error}.\nReponse: {text}")  
        with open(error_log_file, 'a') as f:
            f.write(f"\n\nTop-1 ERROR for: {match_error}.\nReponse: {text}\n\nExtracted_answer: {answers}\n extracted_conf: {confs}\n") 
        answers = None
        confs = None
        
        return answers, confs
    elif confs[0] == 0:
        print(f"\n\nconfs[0] == 0 ERROR: {match_error}.\nReponse: {text}")
        with open(error_log_file, 'a') as f:
            f.write(
                f"\n\nconfs[0] == 0 for: {match_error}.\nReponse: {text}\n\nExtracted_answer: {answers}\n extracted_conf: {confs}\n")
        answers = None
        confs = None
        return answers, confs

    if len(match_error) > 0:
        # If no match is found, print a message
        print(f"\n\nMatch error for: {match_error}.\nReponse: {text}")  
        with open(error_log_file, 'a') as f:
            f.write(f"\n\nMatch error for: {match_error}.\nReponse: {text}\n\nExtracted_answer: {answers}\n extracted_conf: {confs}\n")  

    
    return answers, confs