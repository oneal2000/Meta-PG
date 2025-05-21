# 直接微调
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
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

from accelerate import Accelerator

from basic import ModelCreator
from data import WikiMultiHopQA, HotPotQA, mix_datasets, MixMulti
import ICL
import Meta

get_data = Meta.get_val_data

@dataclass
class FTConfig:
    model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    use_simple_prompt: bool = False
    blind_context: bool = True
    load_dtype: str = "default"
    def __init__(self, args_dict = None):
        if args_dict:
            for k, v in args_dict.items():
                if v is not None:
                    setattr(self, k, v)

class FTWrapper(nn.Module):
    def __init__(self, model, tokenzier, config: FTConfig = FTConfig()):
        super().__init__()
        self.model = model
        self.tokenzier = tokenzier
        self.config = config
    def adapt(self, adapted_named_parameters):
        with torch.no_grad():
            self.backup_named_parameters = {}
            for name, para in self.model.named_parameters():
                if name in adapted_named_parameters:
                    self.backup_named_parameters[name] = para.data.clone().cpu()
                    para.data.copy_(adapted_named_parameters[name])
    def recover(self):
        with torch.no_grad():
            for name, para in self.model.named_parameters():
                if name in self.backup_named_parameters:
                    para.data.copy_(self.backup_named_parameters[name])
        del self.backup_named_parameters
    def forward(self, **kwargs):
        return self.model(**kwargs)
    def derive_state_dict(self):
        return get_peft_model_state_dict(self.model)
    
def load_FTWrapper(dir: str,  random_init=False):
    # random_init的意义：仅仅是一个其它设置一样的，LoRA随机初始化的模型。
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
    ft_config_file = os.path.join(dir, "ft_config.json")
    if os.path.exists(ft_config_file):
        with open(ft_config_file, "r") as f:
            ft_config = FTConfig(json.load(f))
    else:
        ft_config = FTConfig()
    model_name_or_path = ft_config.model_name_or_path
    if ft_config.load_dtype == "default":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    elif ft_config.load_dtype == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
    elif ft_config.load_dtype == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")
    else:
        raise NotImplementedError
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not random_init:
        state_dict_path = os.path.join(dir, "state_dict.pth")
        state_dict = torch.load(state_dict_path)
        set_peft_model_state_dict(model, state_dict)
    return FTWrapper(model, tokenizer, ft_config)

class FTModelCreator(ModelCreator):
    def __init__(self, ft_wrapper: FTWrapper, generation_config=None, blind_context = None, use_simple_prompt = None, device = None):
        super().__init__()
        self.ft_wrapper = ft_wrapper
        self.generation_config = generation_config
        self.blind_context = blind_context
        self.use_simple_prompt = use_simple_prompt
        self.device = device
    def _build(self, contents = None):
        return ICL.ICLModelBox(
            self.ft_wrapper.model, self.ft_wrapper.tokenzier, contents["contents"], self.generation_config,
            blind_context=self.ft_wrapper.config.blind_context if self.blind_context is None else self.blind_context,
            use_simple=self.ft_wrapper.config.use_simple_prompt if self.use_simple_prompt is None else self.use_simple_prompt,
            device=self.device
        )
    def recover(self):
        pass # 没有要恢复的

class FTsimAdaModelCreator(ModelCreator):
    def __init__(self, ft_wrapper: FTWrapper, lr = 2e-4, max_context_len = 2048, generation_config=None, blind_context = None, use_simple_prompt = None, device = None):
        super().__init__()
        self.ft_wrapper = ft_wrapper
        self.lr = lr # 默认值2e-4取自LoRA原论文Table 12
        self.max_context_len = max_context_len
        self.generation_config = generation_config
        self.blind_context = blind_context
        self.use_simple_prompt = use_simple_prompt
        self.device = device
    def _build(self, contents = None):
        context = contents["contents"]["context"]
        task = self.ft_wrapper.tokenzier(
            context,
            truncation=True,
            max_length=self.max_context_len,
            return_tensors="pt"
        )
        task["labels"] = task["input_ids"]
        if self.device:
            task = task.to(self.device)
        
        self.ft_wrapper.model.train()
        loss = compute_loss(self.ft_wrapper, task)
        named_parameters_requires_grad = OrderedDict({
            name: para
            for name, para in self.ft_wrapper.model.named_parameters() 
                if para.requires_grad
        })
        grads = torch.autograd.grad(loss, named_parameters_requires_grad.values())
        del loss

        adapted_named_parameters = OrderedDict()
        with torch.no_grad():
            for (key, val), grad in zip(named_parameters_requires_grad.items(), grads):
                adapted_named_parameters[key] = val - self.lr * grad
        self.ft_wrapper.adapt(adapted_named_parameters)
        del named_parameters_requires_grad
        del adapted_named_parameters
        del grads

        return ICL.ICLModelBox(
            self.ft_wrapper.model, self.ft_wrapper.tokenzier, contents["contents"], self.generation_config,
            blind_context=self.ft_wrapper.config.blind_context if self.blind_context is None else self.blind_context,
            use_simple=self.ft_wrapper.config.use_simple_prompt if self.use_simple_prompt is None else self.use_simple_prompt,
            device=self.device
        )
    def recover(self):
        self.ft_wrapper.recover()

@dataclass
class TrainArgs:
    batch_size: int = 16
    lr: float = 2e-4
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

def compute_loss(model, task):
    return model.forward(**task).loss

def eval(ft_wrapper: FTWrapper, val_set, generation_config, max_num_samples = None, only_related = True, only_loss = False, device = None):
    # TODO: 如果使用多卡，则需要好好处理device的问题。因为并没有好好地将val_set上的数据用accelerator封装起来

    creator = FTModelCreator(ft_wrapper, generation_config, device=device)
    results = val_set.inference(creator, max_num_samples=max_num_samples, only_related=only_related, evaluate_loss=("only" if only_loss else False))
    if "loss" in results:
        logger.info("Loss on val_set: {}".format(results["loss"]))
    if only_loss:
        return results["loss"]
    return val_set.evaluate(results["predictions"])

def train(ft_wrapper: FTWrapper, train_set: Dataset, val_set, train_args: TrainArgs, output_dir, generation_config, get_score = lambda x: x['f1'], logfile=None):
    def collate_fn(batch):
        collated = []
        for data in batch:
            if ft_wrapper.config.blind_context:
                context = None
            else:
                context = data["context"]
                context = ft_wrapper.tokenzier.decode(ft_wrapper.tokenzier.encode(context)[:train_args.max_context_len], skip_special_tokens=True)
            collated.append(
                get_data(
                    tokenizer=ft_wrapper.tokenzier, 
                    qas=data["qas"],
                    context=context,
                    use_simple_prompt=ft_wrapper.config.use_simple_prompt
                )
            )
        return collated
    train_dataloader = DataLoader(train_set, batch_size=train_args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    optimizer = optim.AdamW(
        ft_wrapper.parameters(), 
        lr=train_args.lr,
    ) # TODO: 探索其他优化器

    total_steps = len(train_dataloader) * train_args.num_epochs
    # 预热阶段
    warmup_steps = int(total_steps * 0.1)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    if train_args.lr_scheduler == "CosineAnnealingLR":
        # CosineAnnealingLR 主调度器
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
    elif train_args.lr_scheduler == "LinearLR":
        # LinearLR 主调度器
        main_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=total_steps - warmup_steps,
            start_factor=1.0,
            end_factor=0.0
        )
    # 组合调度器
    lr_scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_steps]
    )

    accelerator = Accelerator(mixed_precision=train_args.mixed_precision)
    ft_wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(ft_wrapper, optimizer, train_dataloader, lr_scheduler)

    steps_count = 0; avg_loss = 0
    best_score = 0; best_results = None; best_steps = 0
    def eval_and_save():
        nonlocal accelerator, best_score, best_results, best_steps, train_args  # 声明使用外部变量
        logger.info("Evaluating ...")
        results = eval(
            ft_wrapper=ft_wrapper, 
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
                state_dict = ft_wrapper.derive_state_dict()
                os.makedirs(output_dir, exist_ok=True)
                accelerator.save(state_dict, os.path.join(output_dir, "state_dict.pth"))
                logger.info("Save state_dict to {}".format(os.path.join(output_dir, "state_dict.pth")))

    optimizer.zero_grad(set_to_none = True)
    # with accelerator.autocast(): # 不要加
    for epoch in range(train_args.num_epochs):
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            batch_loss = 0
            for task in batch:
                loss = compute_loss(ft_wrapper, task) / len(batch)
                batch_loss += loss.item()
                accelerator.backward(loss)
                del loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # 添加梯度裁剪
            # 神奇的是，将梯度裁剪改为accelerator的版本后，效果将会差很多。
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)

            # 更新学习率调度器
            lr_scheduler.step()

            avg_loss += batch_loss

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
        state_dict = model.derive_state_dict()
        os.makedirs(output_dir, exist_ok=True)
        accelerator.save(state_dict, os.path.join(output_dir, "state_dict.pth"))
        logger.info("Save state_dict to {}".format(os.path.join(output_dir, "state_dict.pth")))

    return best_results, best_steps

if __name__ == "__main__":
    '''
    如果本脚本直接被调用，则做好预训练的工作（不负责加载预训练好的PTWrapper）
    '''

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--peft_config_file", default="../config/sample/peft_config.json")
    parser.add_argument("--train_args_file", default="../FT_config/sample/train_args.json")
    parser.add_argument("--generation_config_file", default="../config/sample/generation_config.json")
    parser.add_argument("--ft_config_file", default="../FT_config/sample/ft_config.json")
    parser.add_argument("--output_dir", default="../FT_outputs/sample")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dataset", choices=["2wikimultihopqa", "hotpotqa", "mix_multi"], default="mix_multi")
    args = parser.parse_args()
    
    with open(args.peft_config_file, 'r') as f:
        peft_config = get_peft_config(json.load(f))
    with open(args.train_args_file, 'r') as f:
        train_args = TrainArgs(json.load(f))
    with open(args.generation_config_file, 'r') as f:
        generation_config = GenerationConfig.from_dict(json.load(f))
    with open(args.ft_config_file, 'r') as f:
        ft_config = FTConfig(json.load(f))

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
    with open(os.path.join(args.output_dir, "ft_config.json"), 'w') as f:
        json.dump(ft_config.__dict__, f, indent=4)

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
        # 另外WikiMultiHopQA复制一份，HotPotQA复制两份
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

    model_name_or_path = ft_config.model_name_or_path
    logger.info(f"Loading {model_name_or_path} for FT ...")
    if ft_config.load_dtype == "default":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
    elif ft_config.load_dtype == "fp16":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float16)
    elif ft_config.load_dtype == "auto":
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype="auto")
    else:
        raise NotImplementedError
    model = get_peft_model(model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ft_wrapper = FTWrapper(model, tokenizer, ft_config)

    with open(os.path.join(args.output_dir, "training_log.txt"), 'w') as f:
        results, num_steps = train(
            ft_wrapper=ft_wrapper,
            train_set=train_set,
            val_set=val_set,
            train_args=train_args,
            output_dir=args.output_dir,
            generation_config=generation_config,
            logfile=f
        )
        f.write(f"At step {num_steps} get the best results: {results}\n")
        print(f"At step {num_steps} get the best results: {results}")