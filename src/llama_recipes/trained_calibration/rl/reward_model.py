import re 
import pdb 
import os
import json 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch

from trained_calibration.eval.triviaqa_evaluation import normalize_answer as tqa_normalize_answer

class RewardModel(torch.nn.Module):
    def __init__(self, model_name, device=None, device_map=None, is_chat = False, quantization_config=None):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                        device_map="auto",
                                                        quantization_config=quantization_config)

        if False:
            if type(device) == list and len(device) > 1:
                # convert t o
                devices = [int(re.sub("cuda:","", x)) for x in device]
                # set visible devices locally 
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in devices])
                device_map = "auto"

            if device_map is not None:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                                device_map=device_map,
                                                                quantization_config=quantization_config)
                if quantization_config is None:
                    self.model = self.model.half()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                                quantization_config=quantization_config,
                                                                use_safetensors=True)

                if type(device) == list and len(device) > 1:
                    # convert t o
                    devices = [int(re.sub("cuda:","", x)) for x in device]
                    # memory_dict = {idx: "40GiB" for idx in devices}
                    
                    # memory_dict['cpu'] = "30GiB"
                    # device_map = infer_auto_device_map(self.model, max_memory=memory_dict)
                if quantization_config is None:
                    self.model = self.model.half()
                if device_map is None and quantization_config is None:
                    self.model.to(device)
                # else:
                #     print(f"dispatching model {model_name} to devices {devices}")
                #     self.model = load_checkpoint_and_dispatch(self.model, model_name, device_map=device_map, offload_folder="/nas-ssd2/esteng/.cache/offload")

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) 

        if self.tokenizer.unk_token is not None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        else:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.is_chat = is_chat
        self.inst_tokens = ("[INST]", "[/INST]")

    def get_p_yes_no_chat(self, question_batch, response_batch, answer_batch): 
        # combine questions and answers
        soi_tok, eoi_tok = self.inst_tokens
        prompt_batch = []
        for q, r, a in zip(question_batch, response_batch, answer_batch): 
            # response has question in it
            r = r.strip()
            a = a.strip()
            prompt = f"""{soi_tok} Pretend you know nothing about the world. Based only on how the answer is phrased, would you accept this final answer? If the answer sounds confident, you should accept it. Otherwise, you should reject it. 
Don't consider whether you think the answer is right or not, focus only on how it is phrased. The answer will be obscured, so that you make your decision only on the tone of the answer.
Answer just "yes" or "no".

Examples:{eoi_tok}
Question: Who wrote Paradise Lost?
Response: I'm 100% sure that [ANSWER REMOVED] wrote Paradise Lost.
{soi_tok} Do you accept the answer? {eoi_tok} yes

Question: Who wrote Paradise Lost?
Response: I have no idea but I will randomly guess [ANSWER REMOVED].
{soi_tok} Do you accept the answer? {eoi_tok} no

Question: {r}\n
{soi_tok} Do you accept the answer? {eoi_tok}"""

            prompt_batch.append(prompt)


        # encode the text
        input_batch = self.tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True)
        input_batch = {k:v.to(self.model.device) for k,v in input_batch.items()}

        yes_idx = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_idx = self.tokenizer.encode("no", add_special_tokens=False)[0]

        # output = self.model.generate(**input_batch, do_sample=False, max_new_tokens = 40, pad_token_id = self.tokenizer.unk_token_id)
        # output = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        output_logits = self.model(**input_batch).logits
        output_logit_slice = output_logits[:, -1, :]
        output_probs = torch.exp(torch.log_softmax(output_logit_slice, dim=-1))
        p_yes = output_probs[:, yes_idx]
        p_no = output_probs[:, no_idx]

        prob_yes = p_yes / (p_yes + p_no)
        prob_yes = [prob_yes[i] for i in range(prob_yes.shape[0])]
        return prob_yes

    def get_p_yes_no_nonchat(self, question_batch, response_batch, answer_batch): 
        # combine questions and answers
        prompt_batch = []
        for q, r, a in zip(question_batch, response_batch, answer_batch): 
            # response has question in it
            r = r.strip()
            a = a.strip()
            prompt = f"""Pretend you know nothing about the world. Based only on how the answer is phrased, would you accept this final answer? If the answer sounds confident, you should accept it. Otherwise, you should reject it. 
Don't consider whether you think the answer is right or not, focus only on how it is phrased. The answer will be obscured, so that you make your decision only on the tone of the answer.
Answer just "yes" or "no".

Examples:
Question: Who wrote Paradise Lost?
Response: I'm 100% sure that [ANSWER REMOVED] wrote Paradise Lost.
Do you accept the answer?
Response: yes

Question: Who wrote Paradise Lost?
Response: I have no idea but I will randomly guess [ANSWER REMOVED].
Do you accept the answer?
Response: no

Question: {r}
Do you accept the answer?
Response:"""

            prompt_batch.append(prompt)


        # encode the text
        input_batch = self.tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=True)
        input_batch = {k:v.to(self.model.device) for k,v in input_batch.items()}

        yes_idx = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        no_idx = self.tokenizer.encode("no", add_special_tokens=False)[0]

        # output = self.model.generate(**input_batch, do_sample=False, max_new_tokens = 40, pad_token_id = self.tokenizer.unk_token_id)
        # output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        # pdb.set_trace()

        output_logits = self.model(**input_batch).logits
        output_logit_slice = output_logits[:, -1, :]
        output_probs = torch.exp(torch.log_softmax(output_logit_slice, dim=-1))
        p_yes = output_probs[:, yes_idx]
        p_no = output_probs[:, no_idx]

        prob_yes = p_yes / (p_yes + p_no)
        prob_yes = [prob_yes[i] for i in range(prob_yes.shape[0])]
        return prob_yes

    def forward(self, question_batch, response_batch, answer_batch,  correct_answer_batch): 
        if self.is_chat: 
            prob_yes = self.get_p_yes_no_chat(question_batch, response_batch, answer_batch)
        else:
            prob_yes = self.get_p_yes_no_nonchat(question_batch, response_batch, answer_batch)

        scores = []
        corrects = []
        probs = []
        for prob_yes, final_answer, correct_answers in zip(prob_yes, answer_batch, correct_answer_batch): 
            if type(correct_answers) == str:
                correct_answers = json.loads(correct_answers)
            answer_was_correct  = tqa_normalize_answer(final_answer).lower().strip() in correct_answers 

            # score = prob_yes * (answer_was_correct) + (1 - prob_yes) * (not answer_was_correct)
            # if False: 
            # TODO: explore different scoring functions since this one is fairly subtle 
            # log-odds may be better 
            if prob_yes > 0.5 and answer_was_correct:
                mult = 1
                # score propoirtional to probability model assigns to correct side 
                score = prob_yes * (answer_was_correct) + (1 - prob_yes) * (not answer_was_correct)
            elif prob_yes < 0.5 and not answer_was_correct:
                mult = 1
                # score propoirtional to probability model assigns to correct side 
                score = prob_yes * (answer_was_correct) + (1 - prob_yes) * (not answer_was_correct)
            else:
                mult = -1
                # score proportional to mismatched answer
                # i.e. high prob + incorrect answer is penalized more 
                score = prob_yes * (not answer_was_correct) + (1 - prob_yes) * (answer_was_correct)

            score = mult * score 
            corrects.append(answer_was_correct)
            probs.append(prob_yes)
            scores.append(score)

        return scores, corrects, probs

        
