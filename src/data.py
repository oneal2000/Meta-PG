import json
from tqdm import tqdm
import logging

from torch.utils.data import Dataset

from basic import *

logger = logging.getLogger(__name__)

# 预训练用数据集基类
# {
#     "context": "...",
#     "qas": {
#         ("question0", "answer0"),
#         ("question1", "answer1"),
#         ...
#     }
# }
class TrainingDataset(Dataset):
    def __init__(
        self,
        data,
        max_qas_num = None
    ):
        '''
        `data` should be a list of dicts, each dict should have keys "context", "question", "answer".
        '''
        self.data = data
        if max_qas_num:
            for datum in data:
                datum["qas"] = datum["qas"][:max_qas_num]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def mix_datasets(datasets):
    data = []
    for dataset in datasets:
        data = data + dataset.data
    return TrainingDataset(data)
    
from evaluation.wikimultihop import eval as wikimultihop_evaluate
from evaluation.wikimultihop import eval_detailed as wikimultihop_evaluate_detailed
class WikiMultiHopQA:
    def __init__(
        self,
        gold_file: str,
        alias_file: str,
        data_loaded = None
    ):
        if data_loaded is None:
            logger.info("Loading WikiMultiHopQA dataset from {}.".format(gold_file))
            with open(gold_file, 'r') as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file
        self.alias_file = alias_file
    def get_context(self, datum, only_related = False): 
        context = ""
        if not only_related:
            for i, entry in enumerate(datum["context"]):
                context += f'Passage {i}: ' + ' '.join(entry[1]) + '\n'
        else:
            related = [title for title, _ in datum["supporting_facts"]]
            i = 0
            for entry in datum["context"]:
                if entry[0] in related:
                    context += f'Passage {i}: ' + ' '.join(entry[1]) + '\n'
                    i += 1
        return context
    def inference(
        self, 
        creator: ModelCreator, 
        evaluate_loss=True, # False, True, "only"
        show_current_loss=False, 
        max_num_samples=None, 
        only_related = False
    ):
        '''
        测试模型性能：对于特定的context，生成专用模型，再用它进行测试。
        '''
        predictions = {}; loss = {}
        loss_sum = 0; num_qas = 0
        num_samples = 0
        for datum in tqdm(self.data):
            context = self.get_context(datum, only_related=only_related)
            materials = {
                "contents": {
                    "context": context
                }
            }
            modelbox = creator.build(materials)
            question = datum["question"]
            id = datum["_id"]
            if evaluate_loss != "only":
                prediction = modelbox.predict(question)
                predictions[id] = prediction
            if evaluate_loss:
                answer = datum["answer"]
                cur_loss = modelbox.loss_evaluate({"question": question, "answer": answer})
                loss[id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{id}:", "cur_loss =", cur_loss)
            creator.recover()
            num_samples += 1
            if max_num_samples is not None and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results
    def inference_shift(
        self,
        creator: ModelCreator,
        only_related = False
    ):
        '''
        验证模型确实做到了“知识注入参数”：注入知识是否相关将会对相应的问答有重大影响
        '''
        predictions = {}
        for i in tqdm(range(len(self.data))):
            # i-1的context配合i的问答
            datum = self.data[i]
            datum_shift = self.data[i-1 if i>0 else len(self.data)-1] 
            context = self.get_context(datum_shift, only_related=only_related)
            materials = {
                "contents": {
                    "context": context
                }
            }
            modelbox = creator.build(materials)
            question = datum["question"]
            id = datum["_id"]
            prediction = modelbox.predict(question)
            predictions[id] = prediction
            creator.recover()
        results = {
            "predictions": {"answer": predictions}
        }
        return results
    def evaluate(self, predictions):
        return wikimultihop_evaluate(predictions, self.gold_file, self.alias_file)
    def evaluate_detailed(self, predictions):
        return wikimultihop_evaluate_detailed(predictions, self.gold_file, self.alias_file)
    def derive_training_dataset(self, only_related = False, flatten = None):
        data = []
        for datum in self.data:
            context = self.get_context(datum, only_related=only_related)
            question = datum["question"]
            answer = datum["answer"]
            data.append({
                "context": context,
                "qas": [(question, answer)]
            })
        return TrainingDataset(data)
    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return WikiMultiHopQA(gold_file=self.gold_file, alias_file=self.alias_file, data_loaded=self.data[:max_num_samples])

from evaluation.hotpot import eval as hotpot_evaluate
from evaluation.hotpot import eval_detailed as hotpot_evaluate_detailed
class HotPotQA:
    def __init__(
        self,
        gold_file: str,
        data_loaded = None
    ):
        if data_loaded is None:
            logger.info("Loading HotPotQA dataset from {}.".format(gold_file))
            with open(gold_file, 'r') as f:
                self.data = json.load(f)
            logger.info("Dataset Loaded.")
        else:
            self.data = data_loaded
        self.gold_file = gold_file
    def get_context(self, datum, only_related = False): 
        context = ""
        if not only_related:
            for i, entry in enumerate(datum["context"]):
                context += f'Passage {i}: ' + ''.join(entry[1]) + '\n' # 注意：HotPotQA在这里不用加空格！
        else:
            related = [title for title, _ in datum["supporting_facts"]]
            i = 0
            for entry in datum["context"]:
                if entry[0] in related:
                    context += f'Passage {i}: ' + ''.join(entry[1]) + '\n'
                    i += 1
        return context
    def inference(
        self, 
        creator: ModelCreator, 
        evaluate_loss=True, # False, True, "only"
        show_current_loss=False, 
        max_num_samples=None, 
        only_related = False
    ):
        '''
        测试模型性能：对于特定的context，生成专用模型，再用它进行测试。
        '''
        predictions = {}; loss = {}
        loss_sum = 0; num_qas = 0
        num_samples = 0
        for datum in tqdm(self.data):
            context = self.get_context(datum, only_related=only_related)
            materials = {
                "contents": {
                    "context": context
                }
            }
            modelbox = creator.build(materials)
            question = datum["question"]
            id = datum["_id"]
            if evaluate_loss != "only":
                prediction = modelbox.predict(question)
                predictions[id] = prediction
            if evaluate_loss:
                answer = datum["answer"]
                cur_loss = modelbox.loss_evaluate({"question": question, "answer": answer})
                loss[id] = cur_loss
                loss_sum += cur_loss
                num_qas += 1
                if show_current_loss:
                    print(f"{id}:", "cur_loss =", cur_loss)
            creator.recover()
            num_samples += 1
            if max_num_samples is not None and num_samples == max_num_samples:
                break
        results = {}
        if evaluate_loss:
            results["loss"] = loss_sum / num_qas
        if evaluate_loss != "only":
            results["predictions"] = {"answer": predictions}
        return results
    def inference_shift(
        self,
        creator: ModelCreator,
        only_related = False
    ):
        '''
        验证模型确实做到了“知识注入参数”：注入知识是否相关将会对相应的问答有重大影响
        '''
        predictions = {}
        for i in tqdm(range(len(self.data))):
            # i-1的context配合i的问答
            datum = self.data[i]
            datum_shift = self.data[i-1 if i>0 else len(self.data)-1] 
            context = self.get_context(datum_shift, only_related=only_related)
            materials = {
                "contents": {
                    "context": context
                }
            }
            modelbox = creator.build(materials)
            question = datum["question"]
            id = datum["_id"]
            prediction = modelbox.predict(question)
            predictions[id] = prediction
            creator.recover()
        results = {
            "predictions": {"answer": predictions}
        }
        return results
    def evaluate(self, predictions):
        return hotpot_evaluate(predictions, self.gold_file)
    def evaluate_detailed(self, predictions):
        return hotpot_evaluate_detailed(predictions, self.gold_file)
    def derive_training_dataset(self, only_related = False, flatten = None):
        data = []
        for datum in self.data:
            context = self.get_context(datum, only_related=only_related)
            question = datum["question"]
            answer = datum["answer"]
            data.append({
                "context": context,
                "qas": [(question, answer)]
            })
        return TrainingDataset(data)
    def derive_trunc_dataset(self, max_num_samples=None):
        if max_num_samples is None:
            max_num_samples = len(self.data)
        return HotPotQA(gold_file=self.gold_file, data_loaded=self.data[:max_num_samples])

class MixMulti:
    # 混合2wikimultihopqa和hotpotqa
    def __init__(
        self,
        wiki: WikiMultiHopQA,
        hotpot: HotPotQA
    ):
        self.wiki = wiki
        self.hotpot = hotpot
    def inference(
        self,
        creator: ModelCreator,
        max_num_samples = None, # 每数据集分别最多可以取的个数，不是总体
        only_related = False,
        evaluate_loss = False,
        show_current_loss = False
    ):
        wiki_results = self.wiki.inference(creator, max_num_samples=max_num_samples, only_related=only_related, evaluate_loss=evaluate_loss, show_current_loss=show_current_loss)
        hotpot_results = self.hotpot.inference(creator, max_num_samples=max_num_samples, only_related=only_related, evaluate_loss=evaluate_loss, show_current_loss=show_current_loss)
        combined = {}
        for k in wiki_results.keys():
            combined[k] = {
                "2wiki": wiki_results[k],
                "hotpot": hotpot_results[k]
            }
        return combined
    def inference_shift(
        self,
        creator: ModelCreator,
        only_related = False
    ):
        wiki_results = self.wiki.inference_shift(creator, only_related=only_related)
        hotpot_results = self.hotpot.inference_shift(creator, only_related=only_related)
        combined = {}
        for k in wiki_results.keys():
            combined[k] = {
                "2wiki": wiki_results[k],
                "hotpot": hotpot_results[k]
            }
        return combined
    def evaluate(
        self,
        predictions
    ):
        score_wiki = self.wiki.evaluate(predictions["2wiki"])
        score_hotpot = self.hotpot.evaluate(predictions["hotpot"])
        combined = {
            "2wiki": score_wiki,
            "hotpot": score_hotpot
        }
        for k in score_wiki.keys():
            combined[k] = (score_wiki[k] + score_hotpot[k]) / 2
            # 后面测试时会保证两边样本数相等
        return combined
    def evaluate_detailed(
        self,
        predictions
    ):
        eval_wiki = self.wiki.evaluate_detailed(predictions["2wiki"])
        eval_hotpot = self.hotpot.evaluate_detailed(predictions["hotpot"])
        combined = {
            "2wiki": eval_wiki,
            "hotpot": eval_hotpot
        }
        for k in eval_wiki["metrics"].keys():
            combined[k] = (eval_wiki["metrics"][k] + eval_hotpot["metrics"][k]) / 2
            # 后面测试时会保证两边样本数相等
        return combined
