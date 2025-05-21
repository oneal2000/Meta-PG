import os
import json
import logging
import time

from transformers import GenerationConfig, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, get_peft_config, get_peft_model

from ICL import ICLModelCreator
from Meta import MetaModelCreator, load_Metalearner
from FT import FTModelCreator, load_FTWrapper, FTsimAdaModelCreator
from data import WikiMultiHopQA, HotPotQA, MixMulti

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import random
import numpy as np
import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ICL", "Meta", "FT", "FTsimAda"])
    parser.add_argument("--dataset", choices=["2wikimultihopqa", "hotpotqa", "mix_multi"], default="mix_multi")
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--blind_context", action="store_true") # 注意，如果测试Meta则通常要开启blind_context
    parser.add_argument("--only_related", action="store_true")
    parser.add_argument("--evaluate_loss", action="store_true")
    parser.add_argument("--detailed", action="store_true") # 输出小题分
    parser.add_argument("--shift", action="store_true") # 错配context和question
    parser.add_argument("--random_init", action="store_true") # 仅在FTsimAda时有效
    parser.add_argument("--prediction_file", default=None)
    parser.add_argument("--mixed_precision", default="bf16")
    args = parser.parse_args()

    if args.dataset == "2wikimultihopqa":
        dataset = WikiMultiHopQA("../data/2wikimultihopqa/dev.json", "../data/2wikimultihopqa/id_aliases.json")
    elif args.dataset == "hotpotqa":
        dataset = HotPotQA("../data/hotpotqa/hotpot_dev_distractor_v1.json")
    elif args.dataset == "mix_multi":
        num_samples_for_eval = 1000
        dataset = MixMulti(
            WikiMultiHopQA(gold_file="../data/2wikimultihopqa/dev.json", alias_file="../data/2wikimultihopqa/id_aliases.json").derive_trunc_dataset(num_samples_for_eval),
            HotPotQA(gold_file="../data/hotpotqa/hotpot_dev_distractor_v1.json").derive_trunc_dataset(num_samples_for_eval)
        )
    else:
        raise NotImplementedError

    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    genration_config = GenerationConfig(max_new_tokens=20)
    if args.mode == "ICL":
        creator = ICLModelCreator(
            args.model_name_or_path, 
            generation_config=genration_config,
            blind_context=args.blind_context
        )
    elif args.mode == "Meta":
        learner = load_Metalearner(args.model_name_or_path)
        creator = MetaModelCreator(
            learner, 
            generation_config=genration_config,
            blind_context=args.blind_context,
            device=accelerator.device
        )
    elif args.mode == "FT":
        ft_wrapper = load_FTWrapper(args.model_name_or_path)
        creator = FTModelCreator(
            ft_wrapper,
            generation_config=genration_config,
            blind_context=args.blind_context,
            device=accelerator.device
        )
    elif args.mode == "FTsimAda":
        ft_wrapper = load_FTWrapper(args.model_name_or_path, random_init=args.random_init)
        creator = FTsimAdaModelCreator(
            ft_wrapper,
            generation_config=genration_config,
            blind_context=args.blind_context,
            device=accelerator.device
        )
    else:
        raise NotImplementedError
    
    import time
    start_time = time.perf_counter()
    if args.shift:
        results = dataset.inference_shift(
            creator, 
            only_related=args.only_related
        )
    else:
        results = dataset.inference(
            creator, 
            evaluate_loss=args.evaluate_loss,
            show_current_loss=False, 
            only_related=args.only_related
        )
    run_time = time.perf_counter() - start_time
    results["run_time"] = run_time

    if args.evaluate_loss:
        print("Loss:", results["loss"])
    evaluation = dataset.evaluate(results["predictions"])
    print(evaluation)
    results["scores"] = evaluation
    if args.detailed:
        detailed = dataset.evaluate_detailed(results["predictions"])
        results["detailed"] = detailed
    if args.prediction_file:
        prediction_dir = os.path.dirname(args.prediction_file)
        os.makedirs(prediction_dir, exist_ok=True)
        with open(args.prediction_file, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)