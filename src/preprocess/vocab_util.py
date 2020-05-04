"""
vocab_util.py
 - generate vocab for each condition
 - ref: NYT_util/clean_vocab.py
"""
import os
import pickle
import argparse
import sys
from collections import Counter
sys.path.append("../")
from params import *


def genWordVocab(vocab_data_folder, cond_list, global_word_ft):
    word_cnt = Counter()
    for cond in cond_list:
        with open(os.path.join(vocab_data_folder, str(cond)+".txt"), "r") as fin:
            for line in fin:
                if line.strip() == "":
                    continue
                word, freq = line.strip().split()
                word_cnt[word] += int(freq)
    fout = open(os.path.join(vocab_data_folder, "vocab.txt"), "w")
    for word, freq in word_cnt.most_common():
        if freq < global_word_ft:
            continue
        fout.write(word+" "+str(freq)+"\n")
    fout.close()


def genCondVocab(vocab_data_folder, cond_list):
    """
    @func: generate vocabulary for condition
    """
    with open(os.path.join(vocab_data_folder, "cond_vocab.txt"), "w") as fout:
        fout.write("\n".join([str(cond) for cond in cond_list]))
    

def genConditionalVocab(cond_data_folder, vocab_data_folder, cond_list, word_ft):
    """
    @func: generate vocab in each condition
    """
    for cond in cond_list:
        cond_corpus_fn = os.path.join(cond_data_folder, str(cond)+".txt")
        cond_vocab_fn = os.path.join(vocab_data_folder, str(cond)+".txt")
        command_fn = "run_vocab.sh"
        ref_fn = os.path.join(src_model_folder, command_fn)
        fin = open(ref_fn, "r")
        lines = fin.readlines()
        fin.close()        
        commands = ["SOURCEDIR="+src_model_folder,\
                    "BUILDDIR="+os.path.join(src_model_folder, "build"), \
                    "VOCAB_FILE="+cond_vocab_fn, \
                    "VOCAB_MIN_COUNT="+str(word_ft), \
                    "VERBOSE=2", \
                    "CORPUS="+cond_corpus_fn]
        new_lines = ["\n".join(commands)] + lines[6:]
        with open(command_fn, "w") as fout:
            fout.write("".join(new_lines))
        os.system("chmod 777 "+command_fn)
        os.system("./"+command_fn)
        


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond_data_folder", required=True, default=None, type=str)
    parser.add_argument("--vocab_data_folder", required=True, default=None, type=str)
    parser.add_argument("--data_type", required=True, default=None, type=str)
    parser.add_argument("--word_ft", default=5, type=int,
                        help="word frequency threhold")
    parser.add_argument("--global_word_ft", default=5, type=int,
                        help="global word frequency threshold")
    args = parser.parse_args()

    if args.data_type.lower() == "nyt":
        cond_list = time_list
    elif args.data_type.lower() == "ice":
        cond_list = region_list
    elif args.data_type.lower() == "eu":
        cond_list = domain_list
    else:
        print("Error: unknown data_type {}".format(args.data_type))

    genCondVocab(args.vocab_data_folder, cond_list)
    
    genConditionalVocab(args.cond_data_folder,
                        args.vocab_data_folder,
                        cond_list,
                        args.word_ft)

    genWordVocab(args.vocab_data_folder,
                 cond_list,
                 args.global_word_ft)

    
