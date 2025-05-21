# 基于Meta-Learning算法预训练
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger("accelerate.utils.other").setLevel(logging.ERROR)

from dataclasses import dataclass
import json
import os
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.func import functional_call

import random
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from peft import get_peft_config, get_peft_model

from accelerate import Accelerator

from basic import ModelCreator
from data import WikiMultiHopQA, HotPotQA, mix_datasets, MixMulti
import ICL

def get_train_data(tokenizer, context, max_len=1024, device = None):
    train_data = tokenizer(context, return_tensors="pt", max_length=max_len, truncation=True)
    # train_data = tokenizer(context, return_tensors="pt", max_length=max_len, truncation=True, padding="max_length") # 压力测试
    train_data["labels"] = train_data["input_ids"]
    if device:
        train_data = train_data.to(device)
    return train_data
def get_val_data(tokenizer, qas, device = None, context = None, use_simple_prompt = True):
    l = []
    for question, answer in qas:
        if context:
            if use_simple_prompt:
                prefix = ICL.simple_ICLprompt_context.format(context=context) + ICL.simple_ICLprompt_question.format(question=question)
            else:
                prefix = ICL.ICLprompt_context.format(context=context) + ICL.ICLprompt_question.format(question=question)
        else:
            if use_simple_prompt:
                prefix = ICL.simple_ICLprompt_context_blind + ICL.simple_ICLprompt_question.format(question=question)
            else:
                prefix = ICL.ICLprompt_context_blind + ICL.ICLprompt_question.format(question=question)
        prefix_tokenized = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
        answer_tokenized = tokenizer(answer + "\n", return_tensors="pt", add_special_tokens=False)
        combined = {
            "input_ids": torch.cat([prefix_tokenized["input_ids"], answer_tokenized["input_ids"]], dim=1),
            "attention_mask": torch.cat([prefix_tokenized["attention_mask"], answer_tokenized["attention_mask"]], dim=1),
            "labels": torch.cat([torch.full_like(prefix_tokenized["input_ids"], -100), answer_tokenized["input_ids"]], dim=1)
        }
        l.append(combined)
    from torch.nn.utils.rnn import pad_sequence
    val_data = {
        "input_ids": pad_sequence([item["input_ids"][0] for item in l], batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence([item["attention_mask"][0] for item in l], batch_first=True, padding_value=0),
        "labels": pad_sequence([item["labels"][0] for item in l], batch_first=True, padding_value=-100)
    }
    if device:
        for k, v in val_data.items():
            val_data[k] = v.to(device)
    return val_data
def tokenize_task(tokenizer, data, device = None, use_simple_prompt = True, max_context_len = 1024):
    return {
        "train_data": get_train_data(tokenizer, data["context"], device=device, max_len=max_context_len),
        "val_data": get_val_data(tokenizer, data["qas"], device=device, context=None, use_simple_prompt=use_simple_prompt)
    }

@dataclass
class LearnerConfig:
    base_model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    init_alpha: float = 1e-2
    alpha_mode: str = "tie" # "full", "tie", "same", "fix"
    use_simple_prompt: bool = False
    load_dtype: str = "default"
    def __init__(self, args_dict = None):
        if args_dict:
            for k, v in args_dict.items():
                if v is not None:
                    setattr(self, k, v)

class MetaLearner(nn.Module):
    def __init__(
        self,
        model,
        tokenizer,
        config: LearnerConfig = LearnerConfig(),
    ):
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.config = config

        device = next(self.base_model.parameters()).device
        init_alpha = config.init_alpha
        if config.alpha_mode == "fix":
            self.alpha = init_alpha
        elif config.alpha_mode == "same":
            self.register_parameter("alpha", nn.Parameter(torch.tensor(init_alpha, requires_grad=True, device=device)))
        else:
            self.alpha = OrderedDict()
            for key, val in self.base_model.named_parameters():
                if val.requires_grad:
                    if config.alpha_mode == "tie":
                        alpha_param = nn.Parameter(torch.tensor(init_alpha, requires_grad=True, device=val.device))
                    else:
                        alpha_param = nn.Parameter(torch.full_like(val, init_alpha, requires_grad=True))
                    self.register_parameter('alpha_{}'.format(key.replace('.', '-')), alpha_param)
                    self.alpha[key] = alpha_param
            
    def adapt(self, adapted_named_parameters):
        with torch.no_grad():
            self.backup_named_parameters = {}
            for name, para in self.base_model.named_parameters():
                if name in adapted_named_parameters:
                    self.backup_named_parameters[name] = para.data.clone().cpu()
                    para.data.copy_(adapted_named_parameters[name])
    def recover(self):
        with torch.no_grad():
            for name, para in self.base_model.named_parameters():
                if name in self.backup_named_parameters:
                    para.data.copy_(self.backup_named_parameters[name])
        del self.backup_named_parameters
    def forward(self, adapted_named_parameters=None, **kwargs):
        if adapted_named_parameters is None:
            return self.base_model(**kwargs)
        return functional_call(self.base_model, adapted_named_parameters,
                               args=(), kwargs=kwargs)
    def derive_meta_state_dict(self):
        meta_state_dict = {
            key: val
            # for key, val in self.state_dict().items()
            # state_dict会自动将requires_grad去掉，所以用named_parameters代替之
            for key, val in self.named_parameters()
                if val.requires_grad
        }
        return meta_state_dict

def load_Metalearner(dir: str, device = None):
    peft_config_file = os.path.join(dir, "peft_config.json")
    if os.path.exists(peft_config_file):
        with open(peft_config_file, "r") as f:
            peft_config = get_peft_config(json.load(f))
    else:
        peft_config = get_peft_config({
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
        })
    learner_config_file = os.path.join(dir, "learner_config.json")
    if os.path.exists(learner_config_file):
        with open(learner_config_file, "r") as f:
            learner_config = LearnerConfig(json.load(f))
    else:
        learner_config = LearnerConfig()
    model_name_or_path = learner_config.base_model_name_or_path
    if learner_config.load_dtype == "default":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    elif learner_config.load_dtype == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
    elif learner_config.load_dtype == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")
        # 用bf16是会报错的
        # https://github.com/meta-llama/llama3/issues/80
        # 好像可以通过升级pytorch解决，但当初配了个低版本的pytorch肯定是有原因的，估计是服务器的cuda太旧了。
        # upd: 找到解决办法了
    else:
        raise NotImplementedError
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    learner_path = os.path.join(dir, "meta_state_dict.pth")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(learner_path, map_location=device)
    learner = MetaLearner(model, tokenizer, config=learner_config)
    learner.load_state_dict(state_dict, strict=False)
    return learner
        

class MetaModelCreator(ModelCreator):
    def __init__(self, learner, generation_config=None, blind_context=True, device = None):
        super().__init__()
        self.learner = learner
        self.generation_config = generation_config
        self.blind_context = blind_context
        self.device = device
    def _build(self, contents = None):
        device = (self.device if self.device else next(self.learner.parameters()).device)
        train_data = get_train_data(self.learner.tokenizer, contents["contents"]["context"], device=device, max_len=2048)
        adapted_name_parameters = train_single_task(self.learner, train_data, create_graph=False)
        self.learner.adapt(adapted_name_parameters)
        return ICL.ICLModelBox(self.learner.base_model, self.learner.tokenizer, 
                           contents=contents["contents"], 
                           generation_config=self.generation_config, 
                           blind_context=self.blind_context, 
                           use_simple=self.learner.config.use_simple_prompt,
                           device=self.device)
    def recover(self):
        self.learner.recover()
        # torch.cuda.empty_cache()

@dataclass
class TrainArgs:
    meta_batch_size: int = 16
    meta_lr: float = 2e-4
    num_epochs: int = 4
    logging_steps: int = 40
    eval_steps: int = 200
    do_eval: bool = True
    save_best: bool = True
    num_samples_for_eval: int = None
    eval_only_loss: bool = False
    train_with_only_related: bool = True
    eval_with_only_related: bool = True
    mixed_precision: str = "fp16"
    lr_scheduler: str = "CosineAnnealingLR"
    max_context_len: int = 1024
    def __init__(self, args_dict = None):
        if args_dict:
            for k, v in args_dict.items():
                if v is not None:
                    setattr(self, k, v)
def train_single_task(learner: MetaLearner, train_data, create_graph = True):
    '''
    Calculate adapted_named_parameters
    '''
    learner.train()

    with torch.backends.cuda.sdp_kernel(enable_flash=not create_graph, enable_math=True, enable_mem_efficient=True):
        loss = learner(**train_data).loss
    
    named_parameters_requires_grad = OrderedDict({
        name: para
        for name, para in learner.base_model.named_parameters() 
            if para.requires_grad
    })
    grads = torch.autograd.grad(
        loss, 
        named_parameters_requires_grad.values(),
        create_graph=create_graph
    )

    adapted_named_parameters = OrderedDict()
    if create_graph:
        for (key, val), grad in zip(named_parameters_requires_grad.items(), grads):
            grad_float = grad.float()
            alpha = (learner.alpha.to(grad_float.device) if learner.config.alpha_mode in ["same", "fix"] else learner.alpha[key])
            adapted_named_parameters[key] = val - alpha * grad_float
    else:
        with torch.no_grad():
            for (key, val), grad in zip(named_parameters_requires_grad.items(), grads):
                grad_float = grad.float()
                alpha = (learner.alpha.to(grad_float.device) if learner.config.alpha_mode in ["same", "fix"] else learner.alpha[key])
                adapted_named_parameters[key] = val - alpha * grad_float
    
    return adapted_named_parameters
def compute_loss(learner: MetaLearner, task):
    adapted_named_parameters = train_single_task(learner, task["train_data"])
    outputs = learner.forward( **task["val_data"], adapted_named_parameters=adapted_named_parameters) # 先假设验证集已经打包好
    loss = outputs.loss
    return loss
def eval(learner: MetaLearner, val_set, generation_config, max_num_samples = None, only_related = True, only_loss = False, device = None):
    creator = MetaModelCreator(learner, generation_config, device=device)
    results = val_set.inference(creator, max_num_samples=max_num_samples, only_related=only_related, evaluate_loss=("only" if only_loss else False))
    if "loss" in results:
        logger.info("Loss on val_set: {}".format(results["loss"]))
    if only_loss:
        return results["loss"]
    return val_set.evaluate(results["predictions"])

def train(learner: MetaLearner, train_set: Dataset, val_set, train_args: TrainArgs, output_dir, generation_config, get_score = lambda x: x['f1'], logfile=None):
    def collate_fn(batch):
        return [tokenize_task(learner.tokenizer, data, use_simple_prompt=learner.config.use_simple_prompt, max_context_len=train_args.max_context_len) for data in batch]
    train_dataloader = DataLoader(train_set, batch_size=train_args.meta_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    meta_optimizer = optim.AdamW(
        learner.parameters(), 
        lr=train_args.meta_lr,
    ) # TODO: 探索其他优化器

    total_steps = len(train_dataloader) * train_args.num_epochs
    # 预热阶段
    warmup_steps = int(total_steps * 0.1)
    warmup_scheduler = optim.lr_scheduler.LinearLR(meta_optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    if train_args.lr_scheduler == "CosineAnnealingLR":
        # CosineAnnealingLR 主调度器
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer, 
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
    elif train_args.lr_scheduler == "LinearLR":
        # LinearLR 主调度器
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            meta_optimizer,
            total_iters=total_steps - warmup_steps,
            start_factor=1.0,
            end_factor=0.0
        )
    # 组合调度器
    lr_scheduler = optim.lr_scheduler.SequentialLR(
        meta_optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_steps]
    )

    accelerator = Accelerator(mixed_precision=train_args.mixed_precision)
    learner, meta_optimizer, train_dataloader, lr_scheduler = accelerator.prepare(learner, meta_optimizer, train_dataloader, lr_scheduler)
    
    steps_count = 0; avg_loss = 0
    best_score = 0; best_results = None; best_steps = 0
    def eval_and_save():
        nonlocal accelerator, best_score, best_results, best_steps, train_args  # 声明使用外部变量
        logger.info("Evaluating ...")
        results = eval(
            learner=learner, 
            val_set=val_set, 
            generation_config=generation_config, 
            max_num_samples=train_args.num_samples_for_eval, 
            only_related=train_args.eval_with_only_related,
            only_loss=train_args.eval_only_loss,
            device=accelerator.device
        )
        logger.info("Validation results: {}".format(results))
        if logfile:
            logfile.write(f"Step {steps_count}: Validation results = {results}\n")
        score = (-results if train_args.eval_only_loss else get_score(results))
        if score > best_score:
            best_score = score; best_results = results; best_steps = steps_count
            logger.info("Best score updated. Current best results: {}".format(best_results))
            if train_args.save_best:
                meta_state_dict = learner.derive_meta_state_dict()
                os.makedirs(output_dir, exist_ok=True)
                accelerator.save(meta_state_dict, os.path.join(output_dir, "meta_state_dict.pth"))
                logger.info("Save meta_state_dict to {}".format(os.path.join(output_dir, "meta_state_dict.pth")))

    meta_optimizer.zero_grad(set_to_none = True)
    # with accelerator.autocast(): # 不要加
    for epoch in range(train_args.num_epochs):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            meta_loss = 0
            for task in batch:
                loss = compute_loss(learner, task) / len(batch)
                meta_loss += loss.item()
                accelerator.backward(loss)
                del loss
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=0.5) # 添加梯度裁剪
            # 神奇的是，将梯度裁剪改为accelerator的版本后，效果将会差很多。
            meta_optimizer.step()
            meta_optimizer.zero_grad(set_to_none = True)

            # 更新学习率调度器
            lr_scheduler.step()

            avg_loss += meta_loss

            steps_count += 1

            if steps_count % train_args.logging_steps == 0:
                avg_loss /= train_args.logging_steps
                logger.info("Avg train loss: {}".format(avg_loss))
                if logfile:
                    logfile.write(f"Step {steps_count}: avg_loss = {avg_loss}\n")
                avg_loss = 0

            if train_args.do_eval and steps_count % train_args.eval_steps == 0:
                eval_and_save()

    if train_args.do_eval and train_args.save_best:
        eval_and_save()
    else:
        meta_state_dict = learner.derive_meta_state_dict()
        os.makedirs(output_dir, exist_ok=True)
        accelerator.save(meta_state_dict, os.path.join(output_dir, "meta_state_dict.pth"))
        logger.info("Save meta_state_dict to {}".format(os.path.join(output_dir, "meta_state_dict.pth")))

    return best_results, best_steps

if __name__ == "__main__":
    '''
    如果本脚本直接被调用，则做好预训练的工作（不负责加载预训练好的MetaLearner）
    '''

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_config_file", default="../config/sample/peft_config.json")
    parser.add_argument("--train_args_file", default="../config/sample/train_args.json")
    parser.add_argument("--generation_config_file", default="../config/sample/generation_config.json")
    parser.add_argument("--learner_config_file", default="../config/sample/learner_config.json")
    parser.add_argument("--output_dir", default="../outputs/sample")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset", choices=["2wikimultihopqa", "hotpotqa", "mix_multi"], default="mix_multi")
    args = parser.parse_args()
    
    with open(args.peft_config_file, 'r') as f:
        peft_config = get_peft_config(json.load(f))
    with open(args.train_args_file, 'r') as f:
        train_args = TrainArgs(json.load(f))
    with open(args.generation_config_file, 'r') as f:
        generation_config = GenerationConfig.from_dict(json.load(f))
    with open(args.learner_config_file, 'r') as f:
        learner_config = LearnerConfig(json.load(f))

    os.makedirs(args.output_dir, exist_ok=args.overwrite)
    with open(os.path.join(args.output_dir, "peft_config.json"), 'w') as f:
        peft_config_dict = peft_config.to_dict()
        peft_config_dict["peft_type"] = "LORA" # 原来是PeftType类型
        peft_config_dict["target_modules"] = list(peft_config_dict["target_modules"])
        json.dump(peft_config_dict, f, indent=4)
    with open(os.path.join(args.output_dir, "train_args.json"), 'w') as f:
        json.dump(train_args.__dict__, f, indent=4)
    with open(os.path.join(args.output_dir, "generation_config.json"), 'w') as f:
        json.dump(generation_config.to_dict(), f, indent=4)
    with open(os.path.join(args.output_dir, "learner_config.json"), 'w') as f:
        json.dump(learner_config.__dict__, f, indent=4)

    if args.dataset == "2wikimultihopqa":
        train_set = WikiMultiHopQA(
            gold_file="../data/2wikimultihopqa/train.json", 
            alias_file="../data/2wikimultihopqa/id_aliases.json"
        ).derive_training_dataset(only_related=train_args.train_with_only_related)
        val_set = WikiMultiHopQA(
            gold_file="../data/2wikimultihopqa/dev.json", 
            alias_file="../data/2wikimultihopqa/id_aliases.json"
        )
    elif args.dataset == "hotpotqa":
        train_set = HotPotQA(
            gold_file="../data/hotpotqa/hotpot_train_v1.1.json", 
        ).derive_training_dataset(only_related=train_args.train_with_only_related)
        val_set = HotPotQA(
            gold_file="../data/hotpotqa/hotpot_dev_distractor_v1.json", 
        )
    elif args.dataset == "mix_multi":
        # WikiMultiHopQA复制一份，HotPotQA复制两份
        train_set_2wiki = WikiMultiHopQA(
            gold_file="../data/2wikimultihopqa/train.json", 
            alias_file="../data/2wikimultihopqa/id_aliases.json"
        ).derive_training_dataset(only_related=train_args.train_with_only_related)
        train_set_hotpot = HotPotQA(
            gold_file="../data/hotpotqa/hotpot_train_v1.1.json", 
        ).derive_training_dataset(only_related=train_args.train_with_only_related)
        train_set = mix_datasets([train_set_2wiki, train_set_hotpot, train_set_hotpot])
        val_set = MixMulti(
            WikiMultiHopQA(gold_file="../data/2wikimultihopqa/dev.json", alias_file="../data/2wikimultihopqa/id_aliases.json").derive_trunc_dataset(train_args.num_samples_for_eval),
            HotPotQA(gold_file="../data/hotpotqa/hotpot_dev_distractor_v1.json").derive_trunc_dataset(train_args.num_samples_for_eval)
        )
    else:
        raise NotImplementedError(f"Unknown data type {args.dataset}")

    model_name_or_path = learner_config.base_model_name_or_path
    logger.info(f"Loading {model_name_or_path} for Meta ...")
    if learner_config.load_dtype == "default":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    elif learner_config.load_dtype == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
    elif learner_config.load_dtype == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")
    else:
        # 用bf16是会报错的，好像是pytorch版本的问题
        # https://github.com/meta-llama/llama3/issues/80
        raise NotImplementedError

    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    learner = MetaLearner(model, tokenizer, learner_config)

    with open(os.path.join(args.output_dir, "training_log.txt"), 'w') as f:
        results, num_steps = train(
            learner=learner,
            train_set=train_set,
            val_set=val_set,
            train_args=train_args,
            output_dir=args.output_dir,
            generation_config=generation_config,
            logfile=f
        )
        f.write(f"At step {num_steps} get the best results: {results}\n")
        print(f"At step {num_steps} get the best results: {results}")