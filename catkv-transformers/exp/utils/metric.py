from enum import Enum
import re
import string
import jieba

from fuzzywuzzy import fuzz
from rouge_score import rouge_scorer
import collections

class MetricType(Enum):
    F1 = "f1"
    RL = "rl"
    RLZH = "rlzh"
    CS = "cs"

# Define a class to calculate the score for each example
def metric_func(metric_type: MetricType, pred, answer, tokenizer):
    if isinstance(answer, list):
        answer = answer[0]

    if metric_type == MetricType.F1:
        return compute_f1(pred, answer, tokenizer)
    elif metric_type == MetricType.RL:
        return compute_rl(pred, answer)
    elif metric_type == MetricType.RLZH:
        return compute_rl_zh(pred, answer)
    elif metric_type == MetricType.CS:
        return code_sim_score(pred, answer)

def code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return (fuzz.ratio(prediction, ground_truth) / 100)

def compute_em(pred, ground_truth):
    pred = parse_generation(pred)

    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]


    return int(normalize_answer(pred) == normalize_answer(ground_truth))

def compute_f1(pred, ground_truth, tokenizer):
    pred = parse_generation(pred)
    
    if isinstance(ground_truth, list):
        ground_truth = ground_truth[0]

    pred_tokens = normalize_answer(pred).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = collections.Counter(ground_truth_tokens) & collections.Counter(pred_tokens)
    num_same = sum(common.values())
    if len(ground_truth_tokens) == 0 or len(pred_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(ground_truth_tokens == pred_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_rl(pred, gold):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rougeL = scorer.score(gold, pred)['rougeL'].fmeasure
    return rougeL

def compute_rl_zh(prediction, ground_truth):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = compute_rl(prediction, ground_truth)
    return score

def parse_generation(s):
    # Strip whitespace, convert to lower case, and get the first word
    first_word = s.strip().lower().split()[0] if s.strip() else ""
    
    if first_word == "yes":
        return "Yes"
    elif first_word == "no":
        return "No"
    else:
        return s

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    # remove the "\n" at the beginning
    s = s.lstrip('\n')
    s = s.split('\n')[0]
    return white_space_fix(remove_articles(remove_punc(lower(s))))