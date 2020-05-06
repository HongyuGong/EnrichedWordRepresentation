"""
eval_metrics.pu
 - evaluation metrics: MRR, MP
"""

import csv
import sys


def evalMRR(query_target_gold, query_target_algo):
    #query_target_gold = readTestData(test_fn)
    sum_MRR = 0
    sum_count = 0
    missing_count = 0
    for query_word in query_target_gold:
        gold_neighbor_list = query_target_gold[query_word]
        if (query_word not in query_target_algo):
            #print("query word not exist:", query_word)
            missing_count += 1
            continue
        for gold_neighbor in gold_neighbor_list:
            sum_count += 1
            gold_cond = gold_neighbor.split("-")[-1]
            target_neighbors = query_target_algo[query_word][gold_cond]
            if (gold_neighbor not in target_neighbors):
                continue
            rank = target_neighbors.index(gold_neighbor) + 1
            sum_MRR += 1.0/rank
    MRR = sum_MRR / float(sum_count)
    print("missing count: {}".format(missing_count))
    print("MRR: {}, sum_count: {}".format(MRR, sum_count))
    return MRR


def evalMP(query_target_gold, query_target_algo, topk):
    # topk: 1, 3, 5, 10
    sum_prec = 0
    sum_count = 0
    missing_count = 0
    for query_word in query_target_gold:
        gold_neighbor_list = query_target_gold[query_word]
        if (query_word not in query_target_algo):
            #print("missing query word: {}".format(query_word))
            missing_count += 1
            continue
        for gold_neighbor in gold_neighbor_list:
            sum_count += 1
            gold_cond = gold_neighbor.split("-")[-1]
            target_neighbors = query_target_algo[query_word][gold_cond][:topk]
            if (gold_neighbor in target_neighbors):
                sum_prec += 1.0
    mp = float(sum_prec) / sum_count
    print("MP@{}: {}, sum_count: {}".format(topk, mp, sum_count))
    return mp
