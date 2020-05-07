"""
cooccur_util.py
 - count word co-occurrences from preprocessed conditional corpora
 - ref: NTY_util/batch_cooccur.py
"""

import os
import numpy as np
import argparse
import sys
from config.params import *


def calcScale(cond_data_folder, cond_list):
    size_list = []
    for cond in cond_list:
        fn = os.path.join(cond_data_folder, str(cond)+".txt")
        fn_size = os.path.getsize(fn)
        size_list.append(fn_size)
    median_size = np.median(size_list)
    scale_list = [float(median_size) / s for s in size_list]
    return scale_list

def countCooccur(cond_data_folder, vocab_data_folder, cooccur_folder,
                 cond_list=[], window_size=5):
    scale_list = calcScale(cond_data_folder, cond_list)
    vocab_fn = os.path.join(vocab_data_folder, "vocab.txt")
    cooccur_fn = os.path.join(cooccur_folder, "cooccur.bin")
    for (cond_ind, cond) in enumerate(cond_list):
        scale = scale_list[cond_ind]
        corpus_fn = os.path.join(cond_data_folder, str(cond)+".txt")
        command_fn = "run_cooccur.sh"
        ref_fn = os.path.join(src_model_folder, command_fn)
        # customize .sh to count co-occurrences
        fin = open(ref_fn, "r")
        lines = fin.readlines()
        fin.close()
        commands = ["CORPUS="+corpus_fn, \
                    "COOCCURRENCE_FILE="+str(cooccur_fn), \
                    "COND="+str(cond_ind+1), \
                    "SCALE="+str(scale), \
                    "SOURCEDIR="+src_model_folder, \
                    "BUILDDIR="+os.path.join(src_model_folder, "build"), \
                    "VOCAB_FILE="+vocab_fn, \
                    "VOCAB_MIN_COUNT=1", \
                    "VERBOSE=2", \
                    "WINDOW_SIZE="+str(window_size)]
        new_lines = ["\n".join(commands), "\n"] + lines[len(commands):]
        with open(command_fn, "w") as fout:
            fout.write("".join(new_lines))
        os.system("chmod 777 "+command_fn)
        os.system("./"+command_fn)


def shufCooccur(cooccur_folder):
    command_fn = "run_cooccur_shuf.sh"
    cooccur_fn = os.path.join(cooccur_folder, "cooccur.bin")
    cooccur_shuf_fn = os.path.join(cooccur_folder, "cooccur_shuf.bin")
    commands = ["COOCCURRENCE_FILE="+str(cooccur_fn), \
                "COOCCURRENCE_SHUF_FILE="+str(cooccur_shuf_fn), \
                "SOURCEDIR="+src_model_folder, \
                "BUILDDIR="+os.path.join(src_model_folder, "build"), \
                "VERBOSE=2", "MEMORY=4.0", \
                "gcc $SOURCEDIR/shuffle.c -o $BUILDDIR/shuffle -lm -pthread "+ \
                 "-ffast-math -march=native -funroll-loops -Wno-unused-result", \
                "$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE "+ \
                 "< $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"]
    with open(command_fn, "w") as fout:
        fout.write("\n".join(commands))
    os.system("chmod 777 "+command_fn)
    os.system("./"+command_fn)
    
        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond_data_folder", required=True, default=None, type=str)
    parser.add_argument("--vocab_data_folder", required=True, default=None, type=str)
    parser.add_argument("--cooccur_folder", required=True, default=None, type=str)
    parser.add_argument("--data_type", required=True, default=None, type=str)
    parser.add_argument("--window_size", default=5, type=int)
    args = parser.parse_args()

    if args.data_type.lower() == "nyt":
        cond_list = time_list
    elif args.data_type.lower() == "ice":
        cond_list = region_list
    elif args.data_type.lower() == "eu":
        cond_list = domain_list
    else:
        print("Error: unknown data_type {}".format(args.data_type))

    if not os.path.isdir(args.cooccur_folder):
        os.makedirs(args.cooccur_folder)
        
    countCooccur(args.cond_data_folder,
                 args.vocab_data_folder,
                 args.cooccur_folder,
                 cond_list,
                 args.window_size)

    shufCooccur(args.cooccur_folder)
