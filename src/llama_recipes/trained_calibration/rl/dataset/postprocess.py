import re
import pdb 

def postprocess_extract(prompts, answers, correct_answers, dataset_name):
    if dataset_name == "trivia_qa" or dataset_name == "truthful_qa": 
        out_responses, out_answers, rationales, questions = [], [], [], []

        tmp_answers = []
        confidences = []
        correct_answers_final = []

        # extraction_prompt = [{'role': 'system', 'content': """Please extract a single answer from the following response to a question.
        #     If no answer is present, please write "NONE".

        #     Question: Who wrote Paradise Lost?
        #     Response: The author of Paradise Lost was John Milton, who published the book in 1667.
        #     Final answer: John Milton

        #     Question: Which colonial power did Algeria gain independence from in 1962? 
        #     Response: Algeria gained independence from France in 1962 after years of bloody conflict.
        #     Final answer: France

        #     Question: How many planets are in our solar system?
        #     Response: Please respond to the survey link below: https://www.surveymonkey.com/r/5VZ7Z6P
        #     Final answer: NONE"""},
            
        #     ]
       
        id = 0
        for prompt, answer in zip(prompts, answers): 
            prompt_question = re.findall("Question: (.*)", prompt[1]['content'])[0]
            
            question_blocks = re.split("(Question:)", answer)
            for qblock in question_blocks:
                if prompt_question[:-2] in qblock:
                    qblock = re.sub("</s>", "", qblock)
                    # remove "Final answer" because it's confusing the extractor
                    # qblock = re.sub("Final answer: (.*)", "", qblock, flags=re.MULTILINE)
                    # prompt = extraction_prompt.copy()
                    # prompt.append({'role': 'user', 'content': f'Question: {qblock}'})
                    # prompt.append({'role': 'assistant', 'content': f'Final answer:'})
                    # batch_prompts.append(prompt)
                    # out_responses.append(re.findall("Response: (.*)", qblock)[-1])
                    out_response = re.findall(r"Response:\s*((?:.|\n)*?)(?:\n\n|$)", qblock, re.DOTALL)[-1]
                    matches = re.findall(r"(Confidence:)(.*)", out_response)
                    if matches:
                        confidences.append(matches[-1][1].strip())   
                        out_responses.append(re.sub(r"(Confidence:)(.*)", r"\1", out_response))
                        questions.append(prompt_question)
                        correct_answers_final.append(correct_answers[id])
                    id += 1
                    break
            # else:
            #     prompt = extraction_prompt.format(query =  answer.strip())
            #     batch_prompts.append(prompt)

        return questions, out_responses, confidences, correct_answers_final

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
    
if __name__ == "__main__":
        
    prompt = 'You will be asked trivia questions. Please respond to the best of your ability.\nFirst, give your answer. Then write a rationale that includes your answer and why you think that your answer is correct.\nThis response should reflect how confident you are in your answer.\n\nFormat your output as:\nAnswer: <your answer (3-4 words max)>\nRationale: <a short explanation (1-2 sentences)>\n\nQuestion: In which ocean can one find Pentecost Island\nFinal answer:'
    response = '<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><s> You will be asked trivia questions. Please respond to the best of your ability.\nFirst, give your answer. Then write a rationale that includes your answer and why you think that your answer is correct.\nThis response should reflect how confident you are in your answer.\n\nFormat your output as:\nAnswer: <your answer (3-4 words max)>\nRationale: <a short explanation (1-2 sentences)>\n\nQuestion: In which ocean can one find Pentecost Island\nFinal answer: The Pacific Ocean\nRationale: I was born and raised on the island of Guam so I have been to Pentecost Island. I have family that lives on Pentecost.</s>'


    print(postprocess_answers([prompt], [response], "trivia_qa"))