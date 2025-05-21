import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    return em, f1, prec, recall # 改：原本不返回f1

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

# 修改以让评估脚本只评测出现在predictions中的答案
# 另外只保存了对答案评估的部分，忽视对证据的评估
def eval(prediction, gold_file):
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    
    num_answers = 0
    for dp in gold:
        cur_id = dp['_id']
        if cur_id not in prediction['answer']:
            # print('missing answer {}'.format(cur_id))
            pass
        else:
            num_answers += 1
            update_answer(metrics, prediction['answer'][cur_id], dp['answer'])

    # N = len(gold)
    N = num_answers
    # for k in metrics.keys():
    #     metrics[k] /= N
    for k in metrics.keys():
        metrics[k] = round(metrics[k] / N * 100, 2) # 跟2wiki保持一致

    # print(metrics)
    return metrics

# 加一个带上小题分的评估
def eval_detailed(prediction, gold_file):
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0}
    metrics_detailed = {}
    
    num_answers = 0
    for dp in gold:
        cur_id = dp['_id']
        if cur_id not in prediction['answer']:
            # print('missing answer {}'.format(cur_id))
            pass
        else:
            num_answers += 1
            metrics_detailed[cur_id] = update_answer(metrics, prediction['answer'][cur_id], dp['answer'])

    # N = len(gold)
    N = num_answers
    # for k in metrics.keys():
    #     metrics[k] /= N
    for k in metrics.keys():
        metrics[k] = metrics[k] / N * 100 # 跟2wiki保持一致

    # print(metrics)
    return {
        "metrics": metrics,
        "detailed": metrics_detailed
    }

if __name__ == '__main__':
    eval(sys.argv[1], sys.argv[2])