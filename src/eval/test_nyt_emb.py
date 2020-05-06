"""
test_nyt_emb.py
 - evaluate temporal nyt embeddings on alignment datasets
"""

import argparse
import os
import sys
import pickle
from collections import defaultdict
import numpy as np
import csv
from gensim.models import KeyedVectors
from eval_data_util import readQueryWords
from train.enriched_word_emb import EnrichedWordEmb
from config.params import *


if not os.path.isdir(nyt_eval_res_folder):
    os.makedirs(nyt_eval_res_folder)

def saveTemporalEmb(model):
    for time in time_list:
        model.genEnrichedEmbed(time)
        print("Finish enriched embedding for year: {}".format(time))


def findTemporalNeighbors(model, src_words, topn=10):
    for src_word in src_words:
        res_fn = os.path.join(nyt_eval_res_folder, src_word+".pkl")
        if os.path.isfile(res_fn):
            print("{}.pkl exsits: skip it".format(src_word))
            continue
        temporal_nebs = dict()
        temporal_nebs[src_word] = defaultdict(dict)
        word, src_cond = src_word.strip().split("-")
        src_cond = int(src_cond)
        for trg_cond in time_list:
            if trg_cond == src_cond:
                continue
            neb_dict = model.findInterNeighbors([word], src_cond, trg_cond, topn)
            nebs = neb_dict[word]
            temporal_nebs[src_word][trg_cond] = sorted(nebs, key=lambda tup: tup[1], reverse=True)
        with open(res_fn, "wb") as handle:
            pickle.dump(temporal_nebs, handle)

        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=1)
    
    # for debugging: parallel evaluation
    parser.add_argument("--test", default=False, action="store_true")
    # for debugging: parallel evaluation
    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=100)
    
    # enriched embedding model
    parser.add_argument("--use_cxt_vector", default=False, action="store_true")
    parser.add_argument("--remove_mean", default=True, action="store_true")
    parser.add_argument("--use_cond_word_vocab", default=False, action="store_true")

    args = parser.parse_args()
    dataset = args.dataset
    is_test = args.test
    start_ind = args.start_ind
    end_ind = args.end_ind
    use_cxt_vector = args.use_cxt_vector
    remove_mean = args.remove_mean
    use_cond_word_vocab = args.use_cond_word_vocab

    model = EnrichedWordEmb(vocab_fn=os.path.join(nyt_vocab_data_folder, "vocab.txt"),
                            cond_fn=os.path.join(nyt_vocab_data_folder, "cond_vocab.txt"),
                            sep_emb_fn=os.path.join(nyt_embed_folder, sep_emb_prefix+".txt"),
                            dev_emb_fn=os.path.join(nyt_embed_folder, dev_emb_prefix+".txt"),
                            sep_emb_cxt_fn=os.path.join(nyt_embed_folder, sep_emb_cxt_prefix+".txt"),
                            dev_emb_cxt_fn=os.path.join(nyt_embed_folder, dev_emb_cxt_prefix+".txt"),
                            vocab_folder=nyt_vocab_data_folder,
                            embed_folder=nyt_embed_folder,
                            use_cxt_vector=use_cxt_vector,
                            remove_mean=remove_mean,
                            use_cond_word_vocab=use_cond_word_vocab)
    if not is_test:
        saveTemporalEmb(model)
    else:
        test_fn = os.path.join(nyt_eval_data_folder, "Testset", "testset_"+str(dataset)+".csv")
        res_folder = os.path.join(nyt_eval_res_folder, "testset_"+str(dataset))
        if not os.path.isdir(res_folder):
            os.path.makedirs(res_folder)
        query_words, _ = readQueryWords(test_fn)
        findTemporalNeighbors(model, query_words[start_ind:end_ind], topn=20)
            
        
