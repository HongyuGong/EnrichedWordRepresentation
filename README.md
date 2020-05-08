README

# 1. Download data

## Download corpora

- NYT: Download articles-search-1990-2016.json to data/nyt/raw_data/

- ICE: Obtain the written corpora from nine locations (Canada, East Africa, Hong Kong, India, Ireland, Jamaica, Philippines, Singapore, USA) provided by [International Corpus of English](http://ice-corpora.net/ice/index.html). Save these corpora to data/ice/raw_data/

- eu: Put wikicorpus and reddit coprus to data/eu/cond_data

## Download Testset

- NYT: Download two testsets to data/nyt/eval/

- ICE: The testset is available in data/ice/eval


# 2. Preprocess

- NYT

```bash
python -m preprocess.nyt_data_util
```

Save time-specific corpora to data/nyt/cond_data/[2006].txt
[note] duplicate_text not used for now

- ICE

```bash
python -m preprocess.ice_data_util
```

Save location-specific corpora to data/ice/cond_data/[uk].txt
[note] duplicate_text not used for now


- Euphemism

Directly copy domain-specific corpora to data/eu/cond_data/[wiki.txt][reddit.txt]


# 3. Get vocab

```bash
python -m preprocess.vocab_util 
--cond_data_folder COND_DATA_FOLDER
--vocab_data_folder VOCAB_DATA_FOLDER
--data_type DATA_TYPE
--word_ft WORD_FT
--global_word_ft GLOBAL_WORD_FT
```

* COND_DATA_FOLDER: the folder to save corpora for each condition

* VOCAB_DATA_FOLDER: the folder to save vocabulary

* DATA_TYPE: nyt or ice or eu

* WORD_FT: conditional word frequency threshold, word with frequency higher than the threshold are included in the vocabulary in each condition

* GLOBAL_WORD_FT: global word frequency threshold, word with frequency higher than the threshold are included in the joint vocabulary 


# 4. Count co-occurrences

```bash
python -m preprocess.cooccur_util
--cond_data_folder COND_DATA_FOLDER
--vocab_data_folder VOCAB_DATA_FOLDER
--cooccur_folder COOCCUR_FOLDER
--data_type DATA_TYPE
--window_size WINDOW_SIZE
```

* COND_DATA_FOLDER: the folder to save preprocessed conditional corpora

* VOCAB_DATA_FOLDER: the folder to save vocabulary

* COOCCUR_FOLDER: the folder to save cooccurrence data

* DATA_TYPE: nyt or ice or eu

* EMBED_FOLDER: the folder to save trained embeddings

* EPOCH: 80 for nyt and 40 for ice

* WINDOW_SIZE: the window size to count co-occurring words, e.g., 5 as window size

* co-occurrence file is saved to COOCCUR_FOLDER/

# 5. Learn embedding

```bash
python -m train.train
--cooccur_folder COOCCUR_FOLDER
--vocab_folder VOCAB_FOLDER
--data_type DATA_TYPE
--embed_folder EMBED_FOLDER
--emb_dim 50
--epoch EPOCH
```

* COOCCUR_FOLDER: the folder to save cooccurrence data

* VOCAB_FOLDER: the folder to save vocabulary

* DATA_TYPE: nyt or ice or eu

* EMBED_FOLDER: the folder to save trained embeddings

* EPOCH: 80 for nyt and 40 for ice


* temporal embedding: ewe_temporal.c [ref: glove_region.c]

* spatial embedding: ewe_spatial.c [ref: glove_region_multi_penalty]

* two set of embeddings for condition-independent word embedding and deviation embedding

* only one set of condition embedding

# 6. post-process enriched embedding

```bash
python -m eval.test_nyt_emb --remove_mean --use_cond_word_vocab

python -m eval.test_ice_emb --use_cond_word_vocab
```

* Paramters for post-processing, one can try with or without the following choices. 

--use_cxt_vector: whether to use context word embeddings

--remove_mean: remove mean vector from the set of embeddings in each condition

--use_cond_word_vocab: only generate embeddings for words occurring in the corpus of a given corpus

* save temporal embedding to data/nyt/embed/enriched_[2006].txt

* save spatial embedding to data/ice/embed/enriched_[usa].txt

# 7. test enriched embedding

```bash
python -m eval.test_nyt_emb
--dataset DATASET
--test 
--remove_mean
--use_cond_word_vocab
```

* DATASET: either 1 or 2 which refers to testset_1 or testset_2 in NYT Testset.

* save predictions to data/nyt/eval/result/

```bash
python -m eval.test_ice_emb
--test
--use_cond_word_vocab
```

* save predictions to data/ice/eval/result

```bash
python -m eval.test_eu_emb
--test
--use_cxt_vector[?]
--remove_mean[?]
--use_cond_word_vocab[?]
```

# 8. evaluate embedding on alignment tasks

* On NYT testsets

```bash
python -m eval.eval_emb
--test_fn EVAL_FOLDER/Testset/testset_[1/2].csv
--res_folder EVAL_FOLDER/result/
```

* On ICE testset

```bash
python -m eval.eval_emb
--test_fn ../data/ice/eval/ice_equivalents.txt
--res_folder ../data/ice/eval/result/
```

