import re
import pdb 
import string
import json
from llama_recipes.utils.gpt_answer_scoring import GPTAnswerScoring
import math

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


def extract_number(text):
    # 去掉文本中的逗号
    text = text.replace(',', '')

    # 使用正则表达式提取数字（支持小数和负号）
    match = re.search(r'[-+]?\d*\.\d+|\d+', text)

    if match:
        # 提取到的数值并返回
        return float(match.group(0))
    else:
        # 如果没有找到数字，返回 None
        return None

def confidence_replace(prompts, answers, correct_answers, dataset_name='trivia_qa', vllm=False):
    out_responses, y_None, y, out_confidences, confidences_None, out_response_cleans, questions, correct_answer_cleans = [], [], [], [], [], [], [], []
    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
        answer_scorer = GPTAnswerScoring()
    if True:
        id = 0
        for prompt, answer in zip(prompts, answers):
            if vllm:
                answer = answer.outputs[0].text
                prompt_question = prompt
                question_blocks = [answer]
            else:
                prompt_question = re.findall("Question: (.*)", prompt[1]['content'])[0]
                question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if (prompt_question[:-2] in qblock) or vllm == True :
                    qblock = re.sub("</s>", "", qblock)
                    matches2 = re.findall("Confidence: (.*)", qblock)
                    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
                        response_match = re.search(r'Response:\s*(.*?)(?=\n\s*Confidence:|\n\s*$|$)', qblock, re.DOTALL)
                        if response_match:
                            matches1 = type('obj', (object,), {'group': lambda self: response_match.group(1).strip()})()
                        else:
                            matches1 = re.match(r'^.*?(?=\n|$)', qblock)
                    else:
                        matches1 = re.findall("Final answer: (.*)", qblock)
                    if matches1 and matches2 and (matches2[-1] != ''):
                        out_confidences.append(matches2[-1])  # 如果有匹配，取最后一个
                        confidences_None.append(matches2[-1])
                        if dataset_name == 'gsm8k_dataset':
                            correct = extract_number(matches1[-1]) == json.loads(correct_answers[id])
                        elif dataset_name == "trivia_qa" or dataset_name == "strategy_qa":
                            correct = normalize_answer(matches1[-1]).lower().strip() in json.loads(correct_answers[id])
                        elif dataset_name == "hotpot_qa":
                            correct = answer_scorer.score(prompt_question, matches1.group(), json.loads(correct_answers[id]))
                        elif dataset_name == "truthful_qa":
                            correct = answer_scorer.score(prompt_question, matches1.group(), correct_answers[id])
                        if correct:
                            y.append(1)
                            y_None.append(1)
                        else:
                            y.append(0)
                            y_None.append(0)
                        qblock = re.sub(r"(Confidence: )(\d+%)$", r"\1", qblock, count=1, flags=re.S)
                        if not vllm:
                            prompt[2]['content'] = re.search(r"(Response:.*)", qblock, re.S).group(1)
                        if vllm:
                            out_response_cleans.append(qblock)
                        else:
                            out_response_cleans.append(re.search(r"(Response:.*)", qblock, re.S).group(1))
                        questions.append(prompt_question)
                        correct_answer_cleans.append(json.loads(correct_answers[id]))
                        if vllm:
                            out_responses.append(f"Question: {prompt}\n Response:{qblock}")
                        else:
                            out_responses.append(prompt)
                    else:
                        y_None.append(None)
                        confidences_None.append(None)
                        if vllm:
                            out_responses.append(f"Question: {prompt}\n Response:{qblock}")

            id += 1
    out_confidences2 = []
    for percent_str in out_confidences:
        # 使用正则匹配 "数字%" 格式
        match = re.search(r"(\d+\.?\d*)%", percent_str)
        if match:
            percent = float(match.group(1)) / 100
            out_confidences2.append(percent)
        else:
            # 处理无效值（例如设为 0 或记录警告）
            out_confidences2.append(0.0)
            print(f"Warning: Invalid confidence format: {percent_str}")
    return out_responses, out_response_cleans, questions, out_confidences2, y, y_None, confidences_None, correct_answer_cleans


def confidence_replace_gpt(prompts, answers, correct_answers, dataset_name='trivia_qa', vllm=False):
    out_responses, y_None, y, out_confidences, confidences_None, out_response_cleans, questions, correct_answer_cleans = [], [], [], [], [], [], [], []
    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
        answer_scorer = GPTAnswerScoring()
    if True:
        id = 0
        for prompt, answer in zip(prompts, answers):
            prompt_question = prompt

            qblock = answer

            matches2 = re.findall("Confidence: (.*)", qblock)
            if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
                matches1 = re.match(r'^.*?(?=\n|$)', qblock)
            else:
                matches1 = re.findall("Final answer: (.*)", qblock)
            if matches1 and matches2 and (matches2[-1] != ''):
                out_confidences.append(matches2[-1])  # 如果有匹配，取最后一个
                confidences_None.append(matches2[-1])
                if dataset_name == 'gsm8k_dataset':
                    correct = extract_number(matches1[-1]) == json.loads(correct_answers[id])
                elif dataset_name == "trivia_qa" or dataset_name == "strategy_qa":
                    correct = normalize_answer(matches1[-1]).lower().strip() in json.loads(correct_answers[id])
                elif dataset_name == "hotpot_qa":
                    correct = answer_scorer.score(prompt_question, matches1.group(), json.loads(correct_answers[id]))
                elif dataset_name == "truthful_qa":
                    correct = answer_scorer.score(prompt_question, matches1.group(), correct_answers[id])
                if correct:
                    y.append(1)
                    y_None.append(1)
                else:
                    y.append(0)
                    y_None.append(0)
                qblock = re.sub(r"(Confidence: )(\d+%)$", r"\1", qblock, count=1, flags=re.S)
                out_responses.append(qblock)
                out_response_cleans.append(' ' + qblock)  
                questions.append(prompt_question)  
                correct_answer_cleans.append(json.loads(correct_answers[id])) 
            else:
                y_None.append(None)
                confidences_None.append(None)
                out_responses.append("")

            id += 1
    out_confidences2 = []
    for percent_str in out_confidences:
        # 使用正则匹配 "数字%" 格式
        match = re.search(r"(\d+\.?\d*)%", percent_str)
        if match:
            percent = float(match.group(1)) / 100
            out_confidences2.append(percent)
        else:
            # 处理无效值（例如设为 0 或记录警告）
            out_confidences2.append(0.0)
            print(f"Warning: Invalid confidence format: {percent_str}")
    return out_responses, out_response_cleans, questions, out_confidences2, y, y_None, confidences_None, correct_answer_cleans



def find_logprobs(collection):
    """
    在集合中查找指定token的logprob值
    
    参数:
        collection (dict/set): 包含Logprob对象的集合或字典
        
    返回:
        tuple: (yes_logprob, no_logprob)，未找到则为None
    """
    logprob_yes = None
    logprob_no = None
    
    # 遍历集合元素（假设集合元素是Logprob对象）
    # 如果collection是字典，遍历值：for item in collection.values()
    for item in collection.values():  # 根据你的实际数据结构调整
        if item.decoded_token == ' yes':
            logprob_yes = item.logprob
        elif item.decoded_token == ' no':
            logprob_no = item.logprob
    if logprob_yes and logprob_no:
        return math.exp(logprob_yes) / (math.exp(logprob_yes) + math.exp(logprob_no))        
    else:
        return None

def confidence_replace_yes(prompts, answers, correct_answers, dataset_name='trivia_qa', vllm=False):
    out_responses, y_None, y, out_confidences, confidences_None, out_response_cleans, questions, correct_answer_cleans = [], [], [], [], [], [], [], []
    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
        answer_scorer = GPTAnswerScoring()
    if True:
        id = 0
        for prompt, answer in zip(prompts, answers):
            if vllm:
                response = answer.outputs[0].text
                confidence = answer.outputs[0].logprobs[-2]
                confidence_score = find_logprobs(confidence)
                prompt_question = prompt
                question_blocks = [response]
            else:
                prompt_question = re.findall("Question: (.*)", prompt[1]['content'])[0]
                question_blocks = re.split("(Question:)", response)
            for qblock in question_blocks:
                if (prompt_question[:-2] in qblock) or vllm == True :
                    qblock = re.sub("</s>", "", qblock)
                    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
                        matches1 = re.match(r'^.*?(?=\n|$)', qblock)
                    else:
                        matches1 = re.findall("Final answer: (.*)", qblock)
                    if matches1 and confidence_score:
                        out_confidences.append(confidence_score)  # 如果有匹配，取最后一个
                        confidences_None.append(confidence_score)
                        if dataset_name == 'gsm8k_dataset':
                            correct = extract_number(matches1[-1]) == correct_answers[id]
                        elif dataset_name == "trivia_qa" or dataset_name == "strategy_qa":
                            correct = normalize_answer(matches1[-1]).lower().strip() in json.loads(correct_answers[id])
                        elif dataset_name == "hotpot_qa":
                            correct = answer_scorer.score(prompt_question, matches1.group(), json.loads(correct_answers[id]))
                        elif dataset_name == "truthful_qa":
                            correct = answer_scorer.score(prompt_question, matches1.group(), correct_answers[id])
                        if correct:
                            y.append(1)
                            y_None.append(1)
                        else:
                            y.append(0)
                            y_None.append(0)
                        qblock = re.sub(r"(Confidence: )(\d+%)$", r"\1", qblock, count=1, flags=re.S)
                        if not vllm:
                            prompt[2]['content'] = re.search(r"(Response:.*)", qblock, re.S).group(1)
                        if vllm:
                            out_response_cleans.append(qblock)
                        else:
                            out_response_cleans.append(re.search(r"(Response:.*)", qblock, re.S).group(1))
                        questions.append(prompt_question)
                        if dataset_name == "gsm8k_dataset":
                            correct_answer_cleans.append(correct_answers[id])
                        else:
                            correct_answer_cleans.append(json.loads(correct_answers[id]))
                        if vllm:
                            out_responses.append(f"Question: {prompt}\n Response:{qblock}")
                        else:
                            out_responses.append(prompt)
                    else:
                        y_None.append(None)
                        confidences_None.append(None)
                        if vllm:
                            out_responses.append(f"Question: {prompt}\n Response:{qblock}")

            id += 1
    #out_confidences = [float(percent.strip().strip('%')) / 100 for percent in out_confidences]
    return out_responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans

def confidence_replace_cons(prompts, answers, correct_answers, dataset_name='trivia_qa', vllm=False):
    out_responses, y_None, y, out_confidences, confidences_None, out_response_cleans, questions, correct_answer_cleans = [], [], [], [], [], [], [], []
    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
        answer_scorer = GPTAnswerScoring()
    if True:
        id = 0
        for prompt, answer in zip(prompts, answers):
            correct_list = []
            prompt_question = prompt
            print(prompt_question)
            questions.append(prompt_question)
            for a in answer.outputs:
                response = a.text
                qblock = response

                if vllm == True :
                    qblock = re.sub("</s>", "", qblock)
                    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
                        matches1 = re.match(r'^.*?(?=\n|$)', qblock)
                    else:
                        matches1 = re.findall("Final answer: (.*)", qblock)
                    if matches1:
                        if dataset_name == 'gsm8k_dataset':
                            final_answer = extract_number(matches1[-1])
                            correct = final_answer == correct_answers[id]
                        elif dataset_name == "trivia_qa" or dataset_name == "strategy_qa":
                            final_answer = normalize_answer(matches1[-1]).lower().strip()
                            correct = final_answer in json.loads(correct_answers[id])
                        elif dataset_name == "hotpot_qa":
                            final_answer = matches1.group()
                            correct = answer_scorer.score(prompt_question, final_answer, json.loads(correct_answers[id]))
                        elif dataset_name == "truthful_qa":
                            final_answer = matches1.group()
                            correct = answer_scorer.score(prompt_question, final_answer, correct_answers[id])
                        if correct:
                            y.append(1)
                            y_None.append(1)
                        else:
                            y.append(0)
                            y_None.append(0)
                        out_response_cleans.append(final_answer)
                        out_responses.append(final_answer)
                    else:
                        y_None.append(0)
                        correct_list.append(0)
                        out_responses.append(None)
            id += 1
    return out_responses, out_response_cleans, questions, out_confidences, y, y_None, confidences_None, correct_answer_cleans



def postprocess_extract(prompts, answers, correct_answers, dataset_name='trivia_qa'):
    if dataset_name == "trivia_qa" or dataset_name == "truthful_qa": 
        y, out_confidences = [], []
       
        id = 0
        for prompt, answer in zip(prompts, answers): 
            prompt_question = re.findall("Question: (.*)", prompt[1]['content'])[0]
            question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if prompt_question[:-2] in qblock:
                    qblock = re.sub("</s>", "", qblock)
                    matches1 = re.findall("Final answer: (.*)", qblock)
                    matches2 = re.findall("Confidence: (.*)", qblock)
                    if matches1 and matches2 and (matches2[-1] != ''):
                        out_confidences.append(matches2[-1])  # 如果有匹配，取最后一个
                        if normalize_answer(matches1[-1]).lower().strip() in json.loads(correct_answers[id]):
                            y.append(1)
                        else:
                            y.append(0)
                    else:
                        y.append(None)
                        out_confidences.append(None)  # 如果没有匹配，添加一个默认值（如 None）
            id += 1

        filtered_out_confidences = []
        filtered_y = []
        for confidence, label in zip(out_confidences, y):
            if (confidence is not None) and (label is not None):
                filtered_out_confidences.append(confidence)
                filtered_y.append(label)
        filtered_out_confidences = [float(percent.strip().strip('%')) / 100 for percent in filtered_out_confidences]

        return filtered_out_confidences, filtered_y, y, out_confidences



def postprocess_extract2(prompts, answers, model, tokenizer, correct_answers, dataset_name='trivia_qa'):
    if dataset_name == "trivia_qa" or dataset_name == "truthful_qa": 
        out_responses, out_answers, out_confidences, rationales, questions = [], [], [], [], []

        batch_prompts = []

        extraction_prompt = [{'role': 'system', 'content': """Please extract a single answer from the following response to a question.
            If no answer is present, please write "NONE".

            Question: Who wrote Paradise Lost?
            Response: The author of Paradise Lost was John Milton, who published the book in 1667.
            Final answer: John Milton

            Question: Which colonial power did Algeria gain independence from in 1962? 
            Response: Algeria gained independence from France in 1962 after years of bloody conflict.
            Final answer: France

            Question: How many planets are in our solar system?
            Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
            Final answer: NONE"""},
            
            ]
       

        for prompt, answer in zip(prompts, answers): 
            prompt_question = re.findall("Question: (.*)", prompt[1]['content'])[0]
            questions.append(prompt_question)
            question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if prompt_question[:-2] in qblock:
                    qblock = re.sub("</s>", "", qblock)

                    # remove "Final answer" because it's confusing the extractor
                    qblock = re.sub("Final answer: (.*)", "", qblock, flags=re.MULTILINE)
                    prompt = extraction_prompt.copy()
                    prompt.append({'role': 'user', 'content': f'Question: {qblock}'})
                    prompt.append({'role': 'assistant', 'content': f'Final answer:'})
                    batch_prompts.append(prompt)
                    # try:
                    #     out_responses.append(re.findall("Answer:(.*)", qblock)[-1])
                    # except:
                    #     stop = 1
                    matches = re.findall("Confidence: (.*)", qblock)
                    if matches:
                        out_confidences.append(matches[-1])  # 如果有匹配，取最后一个
                    else:
                        out_confidences.append(None)  # 如果没有匹配，添加一个默认值（如 None）
                    break
            else:
                prompt = extraction_prompt.format(query =  answer.strip())
                batch_prompts.append(prompt)
        # encode the text
        # input_batch = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True)
        input_batch = tokenizer.apply_chat_template(batch_prompts, tokenize=True, padding="longest", truncation=True, return_tensors="pt", continue_final_message=True).to(model.device)
        # try:
        #     input_batch = {k:v.to(model.device) for k,v in input_batch.items()}
        # except AttributeError:
        #     # model is quantized, not needed 
        #     pass 
        generation_kwargs = {
        "min_length": 1,
        "max_new_tokens": 20,
        "top_k": 0.0,
        "top_p": 1.0,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        }
        response_tensors = model.generate(input_batch, **generation_kwargs)
        # output = tokenizer.batch_decode(output, skip_special_tokens=True)
        output = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
        # out_responses_final = []
        for prompt, answer, response in zip(prompts, answers, output):
            try:
                final_answer = re.findall("Final answer: (.*)", response)[-1]
            except IndexError:
                pdb.set_trace()
                final_answer = "NONE"
            out_answers.append(final_answer) 
            # try to anonymize 
            # out_response = re.sub(f"{re.escape(final_answer)}", "[ANSWER REMOVED]", out_response, flags=re.IGNORECASE)
            # out_responses_final.append(out_response)
        
        y=[]
        for out_answer, correct_answer in zip(out_answers, correct_answers):
            if out_answer.lower().strip() in json.loads(correct_answer):
                y.append(1)
            else:
                y.append(0)
        filtered_out_confidences = []
        filtered_y = []
        for confidence, label in zip(out_confidences, y):
            if confidence is not None:
                filtered_out_confidences.append(confidence)
                filtered_y.append(label)
        filtered_out_confidences = [float(percent.strip('%')) / 100 for percent in filtered_out_confidences]

        return filtered_out_confidences, filtered_y

def postprocess_answers(prompts, answers, dataset_name):
    if dataset_name == "trivia_qa": 
        out_responses, out_answers, rationales = [], [], []

        for prompt, answer in zip(prompts, answers): 
            prompt_question = re.findall("Question: (.*)", prompt)[0]
            question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if prompt_question in qblock:
                    qblock = re.sub("</s>", "", qblock)
                    try:
                        try:
                            answer = re.findall("[aA]nswer:([^<]+?)Rationale", qblock, flags=re.DOTALL)[0].strip()
                            rationale = re.findall("Rationale:([^<]+$)", qblock, flags=re.DOTALL)[0].strip()
                        except IndexError:
                            answer = re.findall("[aA]nswer:([^<]+?)Main reasoning", qblock, flags=re.DOTALL)[0].strip()
                            rationale = re.findall("Main reasoning:([^<]+$)", qblock, flags=re.DOTALL)[0].strip()
                        if len(answer) == 0:
                            answer = "NONE"
                        if len(rationale) == 0:
                            rationale = "NONE"
                        out_answers.append(answer)
                        rationales.append(rationale)
                        out_responses.append(qblock)
                        break
                    except IndexError:
                        pass
            else:
                out_responses.append(None)
                out_answers.append("NONE")
                rationales.append("NONE")
        return out_responses, out_answers, rationales
    
def confidence_replace_3level(prompts, answers, correct_answers, dataset_name='trivia_qa', vllm=False):
    out_responses, y_None, y, out_confidences, confidences_None, out_response_cleans, questions, correct_answer_cleans = [], [], [], [], [], [], [], []
    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
        answer_scorer = GPTAnswerScoring()
    if True:
        id = 0
        for prompt, answer in zip(prompts, answers):
            if vllm:
                answer = answer.outputs[0].text
                prompt_question = prompt
                question_blocks = [answer]
            else:
                prompt_question = re.findall("Question: (.*)", prompt[1]['content'])[0]
                question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if (prompt_question[:-2] in qblock) or vllm == True :
                    qblock = re.sub("</s>", "", qblock)
                    matches2 = re.findall("Confidence: (.*)", qblock)
                    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
                        matches1 = re.match(r'^.*?(?=\n|$)', qblock)
                    else:
                        matches1 = re.findall("Final answer: (.*)", qblock)
                    if matches1 and matches2 and (matches2[-1] != ''):
                        out_confidences.append(matches2[-1])  # 如果有匹配，取最后一个
                        confidences_None.append(matches2[-1])
                        if dataset_name == 'gsm8k_dataset':
                            correct = extract_number(matches1[-1]) == json.loads(correct_answers[id])
                        elif dataset_name == "trivia_qa" or dataset_name == "strategy_qa":
                            correct = normalize_answer(matches1[-1]).lower().strip() in json.loads(correct_answers[id])
                        elif dataset_name == "hotpot_qa":
                            correct = answer_scorer.score(prompt_question, matches1.group(), json.loads(correct_answers[id]))
                        elif dataset_name == "truthful_qa":
                            correct = answer_scorer.score(prompt_question, matches1.group(), correct_answers[id])
                        if correct:
                            y.append(1)
                            y_None.append(1)
                        else:
                            y.append(0)
                            y_None.append(0)
                        qblock = re.sub(r"(Confidence: )(\d+%)$", r"\1", qblock, count=1, flags=re.S)
                        if not vllm:
                            prompt[2]['content'] = re.search(r"(Response:.*)", qblock, re.S).group(1)
                        if vllm:
                            out_response_cleans.append(qblock)
                        else:
                            out_response_cleans.append(re.search(r"(Response:.*)", qblock, re.S).group(1))
                        questions.append(prompt_question)
                        correct_answer_cleans.append(json.loads(correct_answers[id]))
                        if vllm:
                            out_responses.append(f"Question: {prompt}\n Response:{qblock}")
                        else:
                            out_responses.append(prompt)
                    else:
                        y_None.append(None)
                        confidences_None.append(None)
                        if vllm:
                            out_responses.append(f"Question: {prompt}\n Response:{qblock}")

            id += 1
    out_confidences2 = []
    for conf_str in out_confidences:
        # Clean the confidence string by removing <|im_end|> and any other special markers
        conf_str = conf_str.lower().strip().replace('<|im_end|>', '').strip()
        conf_str = re.sub(r'[^a-z]', '', conf_str)  # 去除非字母字符
        if conf_str == "high":
            out_confidences2.append(5/6)
        elif conf_str == "medium":
            out_confidences2.append(1/2)
        elif conf_str == "low":
            out_confidences2.append(1/6)
        else:
            # Default to medium if the confidence string is unrecognized
            out_confidences2.append(1/2)
            print(f"Warning: Invalid confidence format: {conf_str}")
    return out_responses, out_response_cleans, questions, out_confidences2, y, y_None, confidences_None, correct_answer_cleans

def confidence_replace_correct(prompts, answers, correct_answers, dataset_name='trivia_qa', vllm=False):
    out_responses, y_None, y, out_confidences, confidences_None, out_response_cleans, questions, correct_answer_cleans = [], [], [], [], [], [], [], []
    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
        answer_scorer = GPTAnswerScoring()
    if True:
        id = 0
        for prompt, answer in zip(prompts, answers):
            if vllm:
                answer = answer.outputs[0].text
                prompt_question = prompt
                question_blocks = [answer]
            else:
                prompt_question = re.findall("Question: (.*)", prompt[1]['content'])[0]
                question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if (prompt_question[:-2] in qblock) or vllm == True :
                    qblock = re.sub("</s>", "", qblock)
                    matches2 = re.findall("Judgement: (.*)", qblock)
                    if dataset_name == "hotpot_qa" or dataset_name == "truthful_qa":
                        matches1 = re.match(r'^.*?(?=\n|$)', qblock)
                    else:
                        matches1 = re.findall("Final answer: (.*)", qblock)
                    if matches1 and matches2 and (matches2[-1] != ''):
                        out_confidences.append(matches2[-1])  # 如果有匹配，取最后一个
                        confidences_None.append(matches2[-1])
                        if dataset_name == 'gsm8k_dataset':
                            correct = extract_number(matches1[-1]) == json.loads(correct_answers[id])
                        elif dataset_name == "trivia_qa" or dataset_name == "strategy_qa":
                            correct = normalize_answer(matches1[-1]).lower().strip() in json.loads(correct_answers[id])
                        elif dataset_name == "hotpot_qa":
                            correct = answer_scorer.score(prompt_question, matches1.group(), json.loads(correct_answers[id]))
                        elif dataset_name == "truthful_qa":
                            correct = answer_scorer.score(prompt_question, matches1.group(), correct_answers[id])
                        if correct:
                            y.append(1)
                            y_None.append(1)
                        else:
                            y.append(0)
                            y_None.append(0)
                        qblock = re.sub(r"(Confidence: )(\d+%)$", r"\1", qblock, count=1, flags=re.S)
                        if not vllm:
                            prompt[2]['content'] = re.search(r"(Response:.*)", qblock, re.S).group(1)
                        if vllm:
                            out_response_cleans.append(qblock)
                        else:
                            out_response_cleans.append(re.search(r"(Response:.*)", qblock, re.S).group(1))
                        questions.append(prompt_question)
                        correct_answer_cleans.append(json.loads(correct_answers[id]))
                        if vllm:
                            out_responses.append(f"Question: {prompt}\n Response:{qblock}")
                        else:
                            out_responses.append(prompt)
                    else:
                        y_None.append(None)
                        confidences_None.append(None)
                        if vllm:
                            out_responses.append(f"Question: {prompt}\n Response:{qblock}")

            id += 1
    out_confidences2 = []
    for conf_str in out_confidences:
        # Clean the confidence string by removing <|im_end|> and any other special markers
        conf_str = conf_str.lower().strip().replace('<|im_end|>', '').strip()
        conf_str = re.sub(r'[^a-z]', '', conf_str)  # 去除非字母字符
        if conf_str == "correct":
            out_confidences2.append(0.6)
        elif conf_str == "incorrect":
            out_confidences2.append(0.2)
        else:
            # Default to medium if the confidence string is unrecognized
            out_confidences2.append(1/2)
            print(f"Warning: Invalid confidence format: {conf_str}")
    return out_responses, out_response_cleans, questions, out_confidences2, y, y_None, confidences_None, correct_answer_cleans


if __name__ == "__main__":
        
    prompt = 'You will be asked trivia questions. Please respond to the best of your ability.\nFirst, give your answer. Then write a rationale that includes your answer and why you think that your answer is correct.\nThis response should reflect how confident you are in your answer.\n\nFormat your output as:\nAnswer: <your answer (3-4 words max)>\nRationale: <a short explanation (1-2 sentences)>\n\nQuestion: In which ocean can one find Pentecost Island\nFinal answer:'
    response = '<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><s> You will be asked trivia questions. Please respond to the best of your ability.\nFirst, give your answer. Then write a rationale that includes your answer and why you think that your answer is correct.\nThis response should reflect how confident you are in your answer.\n\nFormat your output as:\nAnswer: <your answer (3-4 words max)>\nRationale: <a short explanation (1-2 sentences)>\n\nQuestion: In which ocean can one find Pentecost Island\nFinal answer: The Pacific Ocean\nRationale: I was born and raised on the island of Guam so I have been to Pentecost Island. I have family that lives on Pentecost.</s>'


    print(postprocess_answers([prompt], [response], "trivia_qa"))