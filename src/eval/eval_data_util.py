"""
eval_data_util.py
 - prepare evaluation (alignment) data
"""

import csv

def readQueryWords(fn):
    query_words = []
    query_target_dict = dict()
    with open(fn, "rb") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            query_word = row[0].strip()
            target_word = row[1].strip()
            if (query_word not in query_target_dict):
                query_target_dict[query_word] = []
                query_words.append(query_word)
            query_target_dict[query_word].append(target_word)
    return query_words, query_target_dict
