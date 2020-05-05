"""
test_ice_emb.py
 - evaluate spatial ice embeddings on alignment dataset
"""

import argparse
import os
import sys
import pickle
import numpy as np
from gensim.models import KeyedVectors
from collections import defaultdict
from eval_data_util import readQueryWords
sys.path.append("../")
from params import *
sys.path.append("../train/")
from enriched_word_emb import EnrichedWordEmb


if not os.path.isdir(ice_eval_res_folder):
    os.makedirs(ice_eval_res_folder)


def saveSpatialEmb(model):
    for region in region_list:
        model.genEnrichedEmbed(region)


def findSpatialNeighbors(model, src_words, topn=20):
    """
    @return spatial_nebs: {"washington-usa": {"jamaica": [("place-jamaica", 0.6)]}}
    """
    for src_word in src_words:
        res_fn = os.path.join(ice_eval_res_folder, src_word+".pkl")
        if os.path.isfile(res_fn):
            print("{}.pkl exsits: skip it".format(src_word))
            continue
        spatial_nebs = dict()
        spatial_nebs[src_word] = defaultdict(dict)
        word, src_cond = src_word.strip().split("-")
        for trg_cond in region_list:
            if trg_cond == trg_cond:
                continue
            neb_dict = model.findInterNeighbors([word], src_cond, trg_cond, topn)
            nebs = neb_dict[word]
            spatial_nebs[src_word][trg_cond] = sorted(nebs, key=lambda tup: tup[1], reverse=True)
        with open(res_fn, "wb") as handle:
            pickle.dump(spatial_nebs, handle)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--start_ind", type=int, default=0)
    parser.add_argument("--end_ind", type=int, default=100)
    # enriched embedding model
    parser.add_argument("--use_cxt_vector", default=False, action="store_true")
    parser.add_argument("--remove_mean", default=True, action="store_true")
    parser.add_argument("--use_cond_word_vocab", default=False, action="store_true")
    
    args = parser.parse_args()
    is_test = args.test
    start_ind = args.start_ind
    end_ind = args.end_ind
    use_cxt_vector = args.use_cxt_vector
    remove_mean = args.remove_mean
    use_cond_word_vocab = args.use_cond_word_vocab

    model = EnrichedWordEmb(vocab_fn=os.path.join(ice_vocab_data_folder, "vocab.txt"),
                            cond_fn=os.path.join(ice_vocab_data_folder, "cond_vocab.txt"),
                            sep_emb_fn=os.path.join(ice_embed_folder, sep_emb_prefix+".txt"),
                            dev_emb_fn=os.path.join(ice_embed_folder, dev_emb_prefix+".txt"),
                            sep_emb_cxt_fn=os.path.join(ice_embed_folder, sep_emb_cxt_prefix+".txt"),
                            dev_emb_cxt_fn=os.path.join(ice_embed_folder, dev_emb_cxt_prefix+".txt"),
                            vocab_folder=ice_vocab_data_folder,
                            embed_folder=ice_embed_folder,
                            use_cxt_vector=use_cxt_vector,
                            remove_mean=remove_mean,
                            use_cond_word_vocab=use_cond_word_vocab)

    if not is_test:
        saveSpatialEmb(model)
    else:
        fn = os.path.join(ice_eval_data_folder, "ice_equivalents.txt")
        res_folder = os.path.join(ice_eval_res_folder)
        if not os.path.isdir(res_folder):
            os.path.makedirs(res_folder)
        query_words, query_target_dict = readQueryWords(fn)
        findSpatialNeighbors(model, query_words[start_ind:end_ind], topn=20)

    
