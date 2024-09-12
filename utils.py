import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8")]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,2,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def cal_f(preds, labels):
    true_num = 0  # 预测结果中真正的正例数目
    pre_true = 0  # 预测结果中认为是正例的数目
    test_true_num = 0  # 测试集中的正例数目

    # PPI
    for test_node in labels:
        if test_node == 1:
            test_true_num += 1

    for pre_node in preds:
        if pre_node == 1:
            pre_true += 1

    for test_node_child, pre_node_child in zip(labels, preds):
        if test_node_child == pre_node_child and test_node_child == 1 and pre_node_child == 1:
            true_num += 1

    print("-------------------------*", type, "*---------------------------------")
    print("|测试集中的正例的数目: " + str(test_true_num))
    print("|预测结果中认为是正例的数目:" + str(pre_true))
    print("|预测为正例的结果中真正的正例数目:" + str(true_num))
    precision = 0 if pre_true == 0 else true_num / pre_true
    recall = 0 if test_true_num == 0 else true_num / test_true_num
    f1_score = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    precision, recall, f1_score = cal_f(preds, labels)
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1_score
    }


def cal_pr_r_f_ddi(preds, labels):

    print('-----------overall performance-----------')
    print(precision_score(preds, labels, labels=[1, 2, 3, 4], average='micro'))
    print(recall_score(preds, labels, labels=[1, 2, 3, 4], average='micro'))
    print(f1_score(preds, labels, labels=[1, 2, 3, 4], average='micro'))

    print('-----------int-----------')
    print(precision_score(preds, labels, labels=[1], average='micro'))
    print(recall_score(preds, labels, labels=[1], average='micro'))
    print(f1_score(preds, labels, labels=[1], average='micro'))

    print('-----------advise-----------')
    print(precision_score(preds, labels, labels=[2], average='micro'))
    print(recall_score(preds, labels, labels=[2], average='micro'))
    print(f1_score(preds, labels, labels=[2], average='micro'))

    print('-----------mechanism-----------')
    print(precision_score(preds, labels, labels=[3], average='micro'))
    print(recall_score(preds, labels, labels=[3], average='micro'))
    print(f1_score(preds, labels, labels=[3], average='micro'))

    print('-----------effect-----------')
    print(precision_score(preds, labels, labels=[4], average='micro'))
    print(recall_score(preds, labels, labels=[4], average='micro'))
    print(f1_score(preds, labels, labels=[4], average='micro'))