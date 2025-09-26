from openai import AzureOpenAI, OpenAIError, OpenAI
import os
import logging
import time
import string
import re
PROMPT = '''You need to evaluate the correctness of the following LLM response to an answer based on comparison. Please evaluate the correctness of the response with "yes" or "no".
Your response should be completely based on the similarity between LLM response and the correct answer.
DO NOT use your own knowledge base when do the comparison.

Question:
<QUESTION>
LLM Response:
<RESPONSE>
Correct Answer:
<CORRECT_ANSWER>

Your response should use the following format:
Correct: <yes or no>
'''

PROMPT_SAME = '''Given a question and two potential answers, you need to analyze whether the two answers convey the same meaning. Respond with "yes" or "no".
Your response should be completely based on the similarity between the two answers.
DO NOT use your own knowledge base when do the comparison.

Question:
<QUESTION>
LLM Response 1:
<RESPONSE1>
LLM Response 2:
<RESPONSE2>

Your response should use the following format:
Correct: <yes or no>
'''

OTHER_PROMPT = '''For the next task, we are going to evaluate the correctness of the following LLM response to an answer. Please evaluate the correctness of the response on a scale of 1 to 10, where 1 is the least correct and 10 is the most correct.

Question:
<QUESTION>
LLM Response:
<RESPONSE>
Best Answer:
<BEST_ANSWER>
Other correct answers:
<OTHER_CORRECT_ANSWERS>

FIRST provide a one sentence explanation of why you gave the response the score you did.  SECOND, on a NEW LINE, provide only the score you gave the response. Your response should use the following format:
Explanation: <one-sentence explanation>
Score: <score from 1 to 10>
'''

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


class GPTAnswerScoring():
    def __init__(self, prompt=PROMPT, try_times=3):
        self.prompt = PROMPT
        self.prompt_same = PROMPT_SAME
        try:
            self.openai = AzureOpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
                api_version=os.environ['OPENAI_API_VERSION'],
                azure_endpoint=os.environ['OPENAI_AZURE_ENDPOINT'],
            )
        except:
            self.openai = OpenAI(
                api_key=os.environ['OPENAI_API_KEY'],
            )
        self.try_times = try_times
        
    def parse_response(self, response):
        try:
            response = response.split('\n')
            correct = response[0].split(':')[1].strip()
            return correct
        except Exception as e:
            logging.error(e)
            return '', 0
        
    def score(self, question, response, correct_answer):
        prompt = self.prompt.replace('<QUESTION>', question).replace('<RESPONSE>', response).replace('<CORRECT_ANSWER>', correct_answer)
        try_time = 0
        rsp = ""
        while try_time < self.try_times:
            try:
                completion = self.openai.chat.completions.create(
                    model=os.environ['OPENAI_DEPLOYMENT_NAME'],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that helps people find information."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                rsp = completion.choices[0].message.content
                score = self.parse_response(rsp)
                return True if normalize_answer(score) == "yes" else False
            except OpenAIError as e:
                logging.error(e)
                time.sleep(5)
                try_time += 1
        return None
    def score_same(self, question, response1, response2):
        prompt = self.prompt_same.replace('<QUESTION>', question).replace('<RESPONSE1>', response1).replace('<RESPONSE2>', response2)
        try_time = 0
        rsp = ""
        while try_time < self.try_times:
            try:
                completion = self.openai.chat.completions.create(
                    model=os.environ['OPENAI_DEPLOYMENT_NAME'],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that helps people find information."},
                        {"role": "user", "content": prompt}
                    ],
                )
                rsp = completion.choices[0].message.content
                score = self.parse_response(rsp)
                return True if normalize_answer(score) == "yes" else False
            except OpenAIError as e:
                logging.error(e)
                time.sleep(5)
                try_time += 1
        return None
    
    def score_other(self, question, response, best_answer, correct_answers):
        prompt = self.prompt.replace('<QUESTION>', question).replace('<RESPONSE>', response) \
            .replace('<BEST_ANSWER>', best_answer).replace('<OTHER_CORRECT_ANSWERS>', '\n'.join(correct_answers))
        try_time = 0
        while try_time < self.try_times:
            try:
                completion = self.openai.chat.completions.create(
                    model=os.environ['OPENAI_DEPLOYMENT_NAME'],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that helps people find information."},
                        {"role": "user", "content": prompt}
                    ],
                )
                rsp = completion.choices[0].message.content
                explanation, score = self.parse_response(rsp)
                if score < 0 or score > 10:
                    raise OpenAIError('Score out of range')
                return explanation, score
            except OpenAIError as e:
                logging.error(e)
                time.sleep(5)
                try_time += 1
        return "", 0

if __name__ == '__main__':
    gpt_answer_scoring = GPTAnswerScoring()
    explanation, score = gpt_answer_scoring.score('What is the capital of France?', 'The capital of France is Paris.', 'Paris')
    print(explanation, score)
    