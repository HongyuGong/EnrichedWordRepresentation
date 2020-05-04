"""
eval_emb.py
 - evaluate embedding on alignment with ranking metrics
"""

import os
import sys
import pickle
import copy
from eval_data_util import readQueryWords
from eval_metrics import evalMRR, evalMP


def evalEmbAlignment(test_fn, res_folder):
    query_words, query_target_gold = readQueryWords(test_fn)
    query_target_pred = dict()
    for query_word in query_target_gold:
        word_fn = res_folder + query_word + ".pkl"
        if not os.path.isfile(word_fn):
            print("Word not found: {}".format(query_word))
            continue
        with open(word_fn, "rb") as handle:
            word_dict = pickle.load(handle)
        clean_word_dict = dict()
        for cond in word_dict[query_word]:
            clean_word_dict[cond] = [tup[0] for tup in word_dict[query_word][cond]]
        query_target_pred[query_word] = copy.deepcopy(clean_word_dict)
    print("# of GOLD query words: {}, # of PRED query words: {}".format(
        len(query_target_gold), len(query_target_pred)))

    # eval MRR
    evalMRR(query_target_gold, query_target_pred)
    # eval MP
    for topk in [1, 3, 5, 10]:
        evalMP(query_target_gold, query_target_pred, topk)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_fn", default=None, type=str)
    parser.add_argument("--res_folder", default=None, type=str)
    args = parser.parse_args()

    evalEmbAlignment(args.test_fn, args.res_folder)
