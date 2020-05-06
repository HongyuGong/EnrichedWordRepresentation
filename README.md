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

Save time-specific corpora to data/nyt/cond_data/[2006.txt]
[note] duplicate_text not used for now

- ICE

```bash
python -m preprocess.ice_data_util
```

Save location-specific corpora to data/ice/cond_data/[UK.txt]
[note] duplicate_text not used for now


- Euphemism

Directly copy domain-specific corpora to data/eu/cond_data/[wiki.txt][reddit.txt]


# 3. Get vocab

```bash
python -m preprocess.vocab_util 
--cond_data_folder [?]
--vocab_data_folder [?]
--data_type [nyt/ice/eu]
--word_ft [?]
```

* save vocab to data/[nyt]/vocab/[2006.txt]


# 4. Count co-occurrences

```bash
python -m preprocess.cooccur_util
--cond_data_folder [?]
--vocab_data_folder [?]
--cooccur_folder [?]
--data_type [nyt/ice/eu]
--word_ft [?]
--window_size [?]
```

* save to data/[nyt]/cooccur/[nyt_2006.bin]

# 5. Learn embedding

```bash
python -m train.train
--cooccur_folder [?]
--vocab_folder [?]
--data_type [nyt/ice/eu]
--embed_folder [?]
--emb_dim 50
--epoch 40
```

* temporal embedding: ewe_temporal.c [ref: glove_region.c]

* spatial embedding: ewe_spatial.c [ref: glove_region_multi_penalty]

* two set of embeddings for condition-independent word embedding and deviation embedding

* only one set of condition embedding

# 6. post-process enriched embedding

```bash
python -m eval.test_nyt_emb --use_cxt_vector --remove_mean --use_cond_word_vocab

python -m eval.test_ice_emb --remove_mean
```

* Paramters for post-processing

--use_cxt_vector: whether to use context word embeddings

--remove_mean: remove mean vector from the set of embeddings in each condition

--use_cond_word_vocab: only generate embeddings for words occurring in the corpus of a given corpus

* save temporal embedding to data/nyt/embed/enriched_[2006].txt

* save spatial embedding to data/ice/embed/enriched_[usa].txt

# 7. test enriched embedding

```bash
python -m eval.test_nyt_emb
--dataset [1/2] 
--test 
--start_ind 0
--end_ind 20
--use_cxt_vector
--remove_mean
--use_cond_word_vocab
```

* save predictions to data/nyt/eval/result/

```bash
python -m eval.test_ice_emb
--test
--start_ind 0
--end_ind 100
--use_cxt_vector[?]
--remove_mean[?]
--use_cond_word_vocab[?]
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
--test_fn ../data/nyt/eval/Testset/testset_[1/2].csv
--res_folder ../data/nyt/eval/result/
```

* On ICE testset

```bash
python -m eval.eval_emb
--test_fn ../data/ice/eval/ice_equivalents.txt
--res_folder ../data/ice/eval/result/
```

