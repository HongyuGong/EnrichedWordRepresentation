"""
train_util.py
 - train enriched word embeddings
 - EWE_temporal.c, EWE_spatial.c
"""
import os
import sys
import argparse
from config.params import *


def train_embedding(cooccur_folder, vocab_folder, data_type, cond_list, embed_folder,
                    emb_dim=50, epoch=40):
    command_fn = "run_ewe_" + str(data_type) + ".sh"
    commands = ["COOCCURRENCE_SHUF_FILE="+os.path.join(cooccur_folder, "cooccur_shuf.bin"), \
                "VOCAB_FILE="+os.path.join(vocab_folder, "vocab.txt"), \
                "COND_SIZE="+str(len(cond_list)), \
                "DIM="+str(emb_dim), \
                "EPOCH="+str(epoch), \
                "SOURCEDIR="+src_model_folder, \
                "BUILDDIR="+os.path.join(src_model_folder, "build"), \
                "SAVEDIR="+embed_folder]
    if data_type == "nyt":
        commands += [ \
            "gcc $SOURCEDIR/EWE_temporal.c -o $BUILDDIR/EWE_temporal " + \
            "-lm -pthread -ffast-math -march=native -funroll-loops -Wno-unused-result", \
            "$BUILDDIR/EWE_temporal -save-file $SAVEDIR/word_or_cond_vectors "+ \
            "-save-word-cond-file $SAVEDIR/word_and_cond_vectors  " + \
            "-save-context-file $SAVEDIR/word_or_cond_cxt_vectors " + \
            "-save-word-cond-context-file $SAVEDIR/word_and_cond_cxt_vectors " + \
            "-threads 20 -input-file $COOCCURRENCE_SHUF_FILE " + \
            "-alpha 0.75 -x-max 100 -iter $EPOCH -vector-size $DIM -binary 0 " + \
            "-vocab-file $VOCAB_FILE -verbose 2 -model 3 -cond $COND_SIZE"]
    elif data_type == "ice":
        commands += [ \
            "gcc $SOURCEDIR/EWE_spatial.c -o $BUILDDIR/EWE_spatial " + \
            "-lm -pthread -ffast-math -march=native -funroll-loops -Wno-unused-result", \
            "$BUILDDIR/EWE_spatial -save-file $SAVEDIR/word_or_cond_vectors " + \
            "-save-word-cond-file $SAVEDIR/word_and_cond_vectors  " + \
            "-save-context-file $SAVEDIR/word_or_cond_cxt_vectors " + \
            "-save-word-cond-context-file $SAVEDIR/word_and_cond_cxt_vectors " + \
            "-threads 20 -input-file $COOCCURRENCE_SHUF_FILE " + \
            "-alpha 0.75 -x-max 4 -iter $EPOCH -vector-size $DIM -binary 0 " + \
            "-vocab-file $VOCAB_FILE -verbose 2 -model 3 -cond $COND_SIZE"]
    elif data_type == "eu":
        commands += [ \
            "gcc $SOURCEDIR/EWE_temporal.c -o $BUILDDIR/EWE_temporal " + \
            "-lm -pthread -ffast-math -march=native -funroll-loops -Wno-unused-result", \
            "$BUILDDIR/EWE_temporal -save-file $SAVEDIR/word_or_cond_vectors " + \
            "-save-word-cond-file $SAVEDIR/word_and_cond_vectors  " + \
            "-save-context-file $SAVEDIR/word_or_cond_cxt_vectors " + \
            "-save-word-cond-context-file $SAVEDIR/word_and_cond_cxt_vectors " + \
            "-threads 20 -input-file $COOCCURRENCE_SHUF_FILE " + \
            "-alpha 0.75 -x-max 100 -iter $EPOCH -vector-size $DIM -binary 0 " + \
            "-vocab-file $VOCAB_FILE -verbose 2 -model 3 -cond $COND_SIZE"]
    else:
        print("Unknown data_type {}".format(data_type))
        sys.exit(0)
    with open(command_fn, "w") as fout:
        fout.write("\n".join(commands))
    os.system("chmod 777 "+command_fn)
    os.system("./"+command_fn)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cooccur_folder", required=True, default=None, type=str)
    parser.add_argument("--vocab_folder", required=True, default=None, type=str)
    parser.add_argument("--data_type", required=True, default=None, type=str)
    parser.add_argument("--embed_folder", required=True, default=None, type=str)
    parser.add_argument("--emb_dim", default=50, type=int)
    parser.add_argument("--epoch", default=40, type=int)
    args = parser.parse_args()

    if args.data_type == "nyt":
        cond_list = time_list
    elif args.data_type == "ice":
        cond_list = region_list
    elif args.data_type == "eu":
        cond_list = domain_list
    else:
        print("Unknown data_type {}".format(data_type))
        sys.exit(0)

    if not os.path.isdir(args.embed_folder):
        os.makedirs(args.embed_folder)
        
    train_embedding(args.cooccur_folder,
                    args.vocab_folder,
                    args.data_type,
                    cond_list,
                    args.embed_folder,
                    args.emb_dim,
                    args.epoch)
        
