import pickle
import os
import argparse
import re
import sys
sys.path.append("../")
from params import *

# create directories
for folder in [ice_folder, ice_raw_data_folder, ice_cond_data_folder, ice_vocab_data_folder]:
    if not os.path.isdir(folder):
        os.makedirs(folder)

#ice_corpus_folder = "/projects/csl/viswanath/data/hgong6/DynamicEmbedding/Data_ICE/"
#save_folder = "/projects/csl/viswanath/data/hgong6/DynamicEmbedding/ICE_region_slice/orig/"
#dup_save_folder = "/projects/csl/viswanath/data/hgong6/DynamicEmbedding/ICE_region_slice/duplicate/"


 
def tokenizeText(string):
    string = re.sub(r"[^A-Za-z0-9()$,!?\'\`]", " ", string) # add $ to indicate special token
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def cleanText(text_fn, save_fn):
    text_seq = []
    f = open(text_fn, "r")
    for line in f:
        if (line.strip() == ""):
            continue
        word_seq = line.strip().lower().split()
        for word in word_seq:
            if (word.startswith("<") and word.endswith(">")):
                continue
            text_seq.append(word)
    f.close()
    # tokenize text
    text_str = " ".join(text_seq)
    text_str = tokenizeText(text_str)
    # save text
    g = open(save_fn, "a+")
    g.write(text_str+"\n")
    g.close()
    #print("finish processing", text_fn)


def readCanadaCorpus():
    folder = os.path.join(ice_raw_data_folder, "ICE-CAN/Corpus/")
    save_fn = os.path.join(ice_cond_data_folder, "canada.txt")
    for text_fn in os.listdir(folder):
        cleanText(os.path.join(folder, text_fn), save_fn)
    print("done processing Canada corpus...")


def readEastAfricaCorpus():
    folder = os.path.join(ice_raw_data_folder,
                          "ICE East Africa/ICE-EA/corpus/retagged for wsmith/")
    subfolders = [os.path.join(folder, sub) for sub in os.listdir(folder) \
                 if os.path.isdir(os.path.join(folder, sub))]
    save_fn = os.path.join(ice_cond_data_folder, "east_africa.txt")
    for subfolder in subfolders:
        for text_fn in os.listdir(subfolder):
            cleanText(os.path.join(subfolder, text_fn), save_fn)
    print("done processing East Africa Corpus...")

    
def readHKCorpus():
    folder = os.path.join(ice_raw_data_folder, "ICE-HK/CORPUS/")
    save_fn = os.path.join(ice_cond_data_folder, "hk.txt")
    for text_fn in os.listdir(folder):
        cleanText(os.path.join(folder, text_fn), save_fn)
    print("done processing Hong Kong Corpus...")


def readIndiaCorpus():
    folder = os.path.join(ice_raw_data_folder, "ICE India/Corpus/")
    save_fn = os.path.join(ice_cond_data_folder, "india.txt")
    for text_fn in os.listdir(folder):
        cleanText(os.path.join(folder, text_fn), save_fn)
    print("done processing India Corpus...")


def readIrelandCorpus():
    folder = os.path.join(ice_raw_data_folder,
                          "ICE-IRL/ICE-Ireland version 1.2#6DAE.2/ICE-Ireland txt/")
    save_fn = os.path.join(ice_cond_data_folder, "ireland.txt")
    for root, dirs, files in os.walk(folder):
        for f in files:
            text_fn = os.path.join(root, f)
            cleanText(text_fn, save_fn)
    print("done processing Ireland Corpus...")
    

def readJamaicaCorpus():
    folder = os.path.join(ice_raw_data_folder, "ICE-JA/CORPUS/")
    save_fn = os.path.join(ice_cond_data_folder, "jamaica.txt")
    for text_fn in os.listdir(folder):
        cleanText(ps.path.join(folder, text_fn), save_fn)
    print("done processing Jamaica Corpus...")


def readPhilippinesCorpus():
    folder = os.path.join(ice_raw_data_folder, "ICE Philippines/Corpus/")
    save_fn = os.path.join(ice_cond_data_folder, "philippines.txt")
    for text_fn in os.listdir(folder):
        cleanText(os.path.join(folder, text_fn), save_fn)
    print("done processing Philippines Corpus...")


def readSingaporeCorpus():
    folder = os.path.join(ice_raw_data_folder, "ICE SINGAPORE/Corpus/")
    save_fn = os.path.join(ice_cond_data_folder, "singapore.txt")
    for text_fn in os.listdir(folder):
        cleanText(os.path.join(folder, text_fn), save_fn)
    print("done processing Singapore Corpus...")


def readUSACorpus():
    folder = os.path.join(ice_raw_data_folder, "ICE-USA/Corpus/")
    save_fn = os.path.join(ice_cond_data_folder, "usa.txt")
    for text_fn in os.listdir(folder):
        cleanText(os.path.join(folder, text_fn), save_fn)
    print("done processing USA Corpus...")


def preprocessRegionTexts(lang):
    if (lang == "canada"):
        readCanadaCorpus()
    elif (lang == "east_africa"):
        readEastAfricaCorpus()
    elif (lang == "hk"):
        readHKCorpus()
    elif (lang == "india"):
        readIndiaCorpus()
    elif (lang == "ireland"):
        readIrelandCorpus()
    elif (lang == "jamaica"):
        readJamaicaCorpus()
    elif (lang == "philippines"):
        readPhilippinesCorpus()
    elif (lang == "singapore"):
        readSingaporeCorpus()
    elif (lang == "usa"):
        readUSACorpus()
    else:
        print("invalid language!")

# [DISABLED]
def duplicateText():
    # get the file size
    region_size_dict = dict()
    for region in region_list:
        orig_fn = save_folder + region + ".txt"
        region_size_dict[region] = os.path.getsize(orig_fn)
    # max size
    max_size = max(region_size_dict.values())
    print("max size: {}".format(max_size))
    for region in region_list:
        print("region: {}, size: {}".format(region, region_size_dict[region]))
        orig_fn = save_folder + region + ".txt"
        dup_fn = dup_save_folder + region + ".txt"
        tmp_fn_list = []
        tmp_num = int(round(float(max_size)/region_size_dict[region]))
        print("duplicate #:", tmp_num)
        for i in range(tmp_num):
            tmp_fn = dup_save_folder + region + "_" + str(i) + ".txt"
            os.system("cp " + orig_fn + " " + tmp_fn)
            tmp_fn_list.append(tmp_fn)
        # concatenate
        os.system("cat " + " ".join(tmp_fn_list) + " > " + dup_fn)
        for fn in tmp_fn_list:
            os.system("rm " + fn)
    print("done duplicating region slices...")
    


if __name__=="__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--lang", type=str, default="usa")
    #args = parser.parse_args()
    #lang = args.lang

    #"""
    # step 1: corpora in different regions
    for region in region_list:
        preprocessRegionTexts(region)
    #"""

    # step 1.5:
    #duplicateText()

    # step 2
    # run DynamicGlove to count pairwise co-occurrences

    # step 3
    # remove stop words

    # step 4
    # merge (w1, w2, region)



       
