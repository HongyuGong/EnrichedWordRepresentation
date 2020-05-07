import json
import re
import os
import sys
from config.params import *

# create directories
for folder in [nyt_folder, nyt_raw_data_folder, nyt_cond_data_folder, nyt_vocab_data_folder]:
  if not os.path.isdir(folder):
    os.makedirs(folder)


def cleanText(string):
  # replace "www.google.com" as "www google com"
  seq = string.split(" ")
  new_seq = []
  for word in seq:
    if ("www" in word):
      new_seq.append(re.sub("\.", " ", word))
    else:
      new_seq.append(word)
  string = " ".join(new_seq)
  string = re.sub(r"[^A-Za-z0-9().,!?\&\'\`-]", " ", string) # add .-&
  string = re.sub(r"[-\&\.]", "", string) # concatenate characters combined with "." or "-" or "&"
  #string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'s", " s", string)
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
  return string.strip().lower()
    

def readUrlYear(fn=os.path.join(nyt_raw_data_folder, nyt_articles)):
  print("Read {}...".format(fn))
  url_year_dict = dict()
  with open(fn, "r") as handle:
    json_data = json.load(handle)
  for text_obj in json_data:
    url = text_obj["url"]
    try:
      date = text_obj["date"]
      year = date.split("-")[0]
    except:
      print("No date:{}".format(text_obj))
      continue
    url_year_dict[url] = year
  return url_year_dict


def readText(fn=os.path.join(nyt_raw_data_folder, nyt_paragraphs)):
  # url_year_dict
  url_year_dict = readUrlYear()
  # paragraphs
  print("Read {}".format(fn))
  year_text_dict = dict()
  with open(fn, "r") as handle:
      json_data = json.load(handle)
  invalid_year_count = 0
  for text_obj in json_data:
      # paras: a list of strings
      paras = text_obj["paragraphs"]
      clean_paras = [cleanText(para) for para in paras]
      url = text_obj["url"]
      try:
        year = int(url_year_dict[url])
      except:
        print("INVALID url: {}".format(url))
        invalid_year_count += 1
        continue
      if (year not in year_text_dict):
          year_text_dict[year] = []
      year_text_dict[year].extend(clean_paras)
  print("invalid year count:", invalid_year_count)
  
  # save text into year-specific file
  for year in year_text_dict:
      slice_fn = os.path.join(nyt_cond_data_folder, str(year)+".txt")
      f = open(slice_fn, "w")
      f.write("\n".join(year_text_dict[year]))
      f.close()
      print("saving texts in year "+str(year)+"...")

# [DISABLED]
def duplicateText():
  # get the file sizes
  year_range = range(1990, 2017) #range(1990, 2017)
  year_size_dict = dict()
  for year in year_range:
    orig_fn = orig_year_slice_folder + "nyt_" + str(year) + ".txt"
    year_size_dict[year] = os.path.getsize(orig_fn)
  # max size
  max_size = max([year_size_dict[year] for year in year_range])
  print("max size:", max_size)
  for year in year_range:
    print(year, "size:", year_size_dict[year])
    orig_fn = orig_year_slice_folder + "nyt_" + str(year) + ".txt"
    dup_fn = duplicate_year_slice_folder + "nyt_" + str(year) + ".txt"
    tmp_fn_list = []
    # copy tmp files
    tmp_num= int(round(float(max_size)/year_size_dict[year]))
    print("duplicate #:", tmp_num)
    for i in range(tmp_num):
      tmp_fn = duplicate_year_slice_folder+"nyt_"+str(year)+"_"+str(i)+".txt"
      os.system("cp "+orig_fn+" "+tmp_fn)
      tmp_fn_list.append(tmp_fn)
    # concatenate
    os.system("cat "+" ".join(tmp_fn_list)+" > "+dup_fn)
    # delete tmp files
    for fn in tmp_fn_list:
      os.system("rm "+fn)
  print("done duplicating year slices....")
      


if __name__=="__main__":
    # step 1: corpora at different time
    readText()
    # step 1.5: duplicate texts for normalization
    #duplicateText()
    # step 2: 
    # run DynamicGlove to count pairwise co-occurrences
    # step 3:
    # remove stop words
    # step 4:
    # merge (w1, w2, time) counts
    
    
        
        
