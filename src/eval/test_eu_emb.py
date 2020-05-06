"""
test_eu_emb.py
 - test embedding on euphemism
"""

import argparse
import os
import sys
import pickle
import csv
import numpy as np
from gensim.models import KeyedVectors
from collections import defaultdict
from eval_data_util import readQueryWords
from config.params import *
from train.enriched_word_emb import EnrichedWordEmb


if not os.path.isdir(eu_eval_res_folder):
    os.makedirs(eu_eval_res_folder)


def saveEuEmb(model):
    for domain in domain_list:
        model.genEnrichedEmbed(domain)
        print("Finish enriched embedding for year: {}".format(domain))


def findEuNeighbors(model, words, topn=20, res_folder=None):
    """
    @func: enumerate neighbors of words with different senses 
    """
    for src_domain in domain_list:
        for trg_domain in domain_list:
            csvfile = open(os.path.join(res_folder, "{}-to-{}.csv".format(
                str(src_domain), str(trg_domain))), "w")
            csv_writer = csv.writer(csvfile, delimiter=",")
            # src_vec to find neighbors in trg_domain
            if src_domain == trg_domain:
                neb_dict = model.findIntraNeighbors(src_domain, words, topn)
            else:
                neb_dict = model.findInterNeighbors(words, src_domain,
                                                    trg_domain, topn)
            csv_writer.writerow(["Keyword", "Neighbors"])
            for keyword in neb_dict:
                csv_writer.writerow([keyword] + [tup[0].encode('utf-8') for tup in neb_dict[keyword]])
            csvfile.close()


def findChangingWords(model, n_words=20):
    src_domain = "wiki"
    trg_domain = "reddit"
    model.sortWordChanges(src_domain, trg_domain, n_words)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true")
    # model parameters
    parser.add_argument("--use_cxt_vector", default=False, action="store_true")
    parser.add_argument("--remove_mean", default=True, action="store_true")
    parser.add_argument("--use_cond_word_vocab", default=False, action="store_true")

    args = parser.parse_args()
    is_test = args.test

    model = EnrichedWordEmb(vocab_fn=os.path.join(eu_vocab_data_folder, "vocab.txt"),
                            cond_fn=os.path.join(eu_vocab_data_folder, "cond_vocab.txt"),
                            sep_emb_fn=os.path.join(eu_embed_folder, sep_emb_prefix+".txt"),
                            dev_emb_fn=os.path.join(eu_embed_folder, dev_emb_prefix+".txt"),
                            sep_emb_cxt_fn=os.path.join(eu_embed_folder, sep_emb_cxt_prefix+".txt"),
                            dev_emb_cxt_fn=os.path.join(eu_embed_folder, dev_emb_cxt_prefix+".txt"),
                            vocab_folder=eu_vocab_data_folder,
                            embed_folder=eu_embed_folder,
                            use_cxt_vector=args.use_cxt_vector,
                            remove_mean=args.remove_mean,
                            use_cond_word_vocab=args.use_cond_word_vocab)

    if not is_test:
        saveEuEmb(model)
    else:
        # find changing and stable words
        findChangingWords(model, n_words=100)
        
        # words with different senses across domains
        words = ["weed", "molly", "pill", "blow", "speed", "pot",
                     "ecstasy", "blotter", "spice", "hash", "downer"]
        findEuNeighbors(model, words, topn=50, res_folder=eu_eval_res_folder)

        
        

