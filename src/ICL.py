import json
import logging

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from basic import Block, Segment, ModelBox, ModelCreator

logger = logging.getLogger(__name__)

ICLprompt_context = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
####CONTEXT begin####
{context}
####CONTEXT end####
You are a helpful AI assistant who is ready to answer user's question according to the CONTEXT. 
Your answer should be concise, which means it must be a short phrase or a single word, and sentences are not allowed!
<|eot_id|><|start_header_id|>user<|end_header_id|>"""
ICLprompt_context_blind = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant who is ready to answer user's question. 
Your answer should be concise, which means it must be a short phrase or a single word, and sentences are not allowed!
<|eot_id|><|start_header_id|>user<|end_header_id|>""" # 看不到context的版本
ICLprompt_question = '''
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Answer: '''

simple_ICLprompt_context = """## CONTEXT
{context}
(CONTEXT END)
"""
simple_ICLprompt_context_blind = """""" # 看不到context的版本
simple_ICLprompt_question = '''
Question: {question}
Answer: '''

class ICLModelBox(ModelBox):
    def __init__(self, model, tokenizer, contents = None, generation_config = None, blind_context = False, use_simple=False, device = None):
        '''
        contents = {"context": "..."}
        blind_context==True means context unseen.
        '''
        if use_simple:
            prefix = (simple_ICLprompt_context_blind if blind_context else simple_ICLprompt_context.format(**contents))
            infer_template = Block([Segment(prefix), Segment(simple_ICLprompt_question)])
        else:
            prefix = (ICLprompt_context_blind if blind_context else ICLprompt_context.format(**contents))
            infer_template = Block([Segment(prefix), Segment(ICLprompt_question)])
        # infer_template只需要填入question，因为context已经装填完毕。
        super().__init__(infer_template)
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.device = device
    def _predict(self, input: Block, **kwargs):
        device = (self.device if self.device else self.model.device)
        tmp = []
        for i, block in enumerate(input.segments):
            tmp.append(self.tokenizer(block.text, return_tensors="pt", add_special_tokens=(i == 0)))
        tokens = {
            k: torch.cat([item[k] for item in tmp], dim=1).to(device)
            for k in tmp[0].keys()
        }
        # 写成这样麻烦的形式是为了带上attention_mask，不然出警告很烦。
        
        self.model.eval()
        with torch.no_grad():
            # with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # 外面的Accelerator应该要能主动地做好混合精度
                outputs = self.model.generate(
                    **tokens, 
                    stop_strings="\n",
                    tokenizer=self.tokenizer,
                    pad_token_id=self.tokenizer.eos_token_id, # 必须加上，不然警告很烦
                    eos_token_id=self.tokenizer.eos_token_id,
                    generation_config=self.generation_config,
                    **kwargs,
                )
        all_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        answer = all_text.split("Answer:")[-1].split("\n")[0].strip()
        
        return answer
    
    def _loss_evaluate(self, input, answer) -> float:
        device = (self.device if self.device else self.model.device)
        tmp = []
        for i, block in enumerate(input.segments):
            tmp.append(self.tokenizer(block.text, return_tensors="pt", add_special_tokens=(i == 0)))
        tmp.append(self.tokenizer(answer, return_tensors="pt", add_special_tokens=False))
        answer_len = tmp[-1]["input_ids"].shape[1]
        tokens = {
            k: torch.cat([item[k] for item in tmp], dim=1).to(device)
            for k in tmp[0].keys()
        } # 写成这样麻烦的形式是为了带上attention_mask，不然出警告很烦。
        labels = torch.cat([
            torch.full_like(tokens["input_ids"][:, :-answer_len], -100),
            tokens["input_ids"][:, -answer_len:]
            ], dim=1)
        # breakpoint()
        self.model.eval()
        with torch.no_grad():
            # with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = self.model(**tokens, labels=labels)
        return outputs.loss.item()

class ICLModelCreator(ModelCreator):
    def __init__(self, model_name_or_path, generation_config=None, blind_context = False, max_context_len = 2048):
        super().__init__()
        logger.info(f"Loading model {model_name_or_path} for ICL ...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_config = generation_config
        logger.info(f"Model loaded.")
        self.blind_context = blind_context 
        self.max_context_len = max_context_len
    def _build(self, contents = None):
        context = contents["contents"]["context"]
        contents["contents"]["context"] = self.tokenizer.decode(self.tokenizer.encode(context)[:self.max_context_len], skip_special_tokens=True)
        return ICLModelBox(self.model, self.tokenizer, contents["contents"], self.generation_config, self.blind_context) # ICL模型没什么要修改的
    def recover(self):
        pass # 没有要恢复的
    