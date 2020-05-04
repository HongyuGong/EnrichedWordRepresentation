README

1. Download data
- NYT: download articles-search-1990-2016.json to data/nyt/raw_data/
- ICE: download ??? to data/ice/raw_data/
- eu: put wikicorpus and reddit coprus to data/eu/cond_data


2. Preprocess
- NYT
python preprocess/nyt_data_util.py 
* save time-specific corpora to data/nyt/cond_data/[2006.txt]
[note] duplicate_text not used for now

- ICE
python preprocess/ice_data_util.py
* save location-specific corpora to data/ice/cond_data/[UK.txt]
[note] duplicate_text not used for now

- Euphemism
* directly copy domain-specific corpora to data/eu/cond_data/[wiki.txt][reddit.txt]


3. Get vocab
python vocab_util.py 
--cond_data_folder [?]
--vocab_data_folder [?]
--data_type [nyt/ice/eu]
--word_ft [?]
* save vocab to data/[nyt]/vocab/[2006.txt]

3. Count co-occurrences
python cooccur_util.py 
--cond_data_folder [?]
--vocab_data_folder [?]
--cooccur_folder [?]
--data_type [nyt/ice/eu]
--word_ft [?]
--window_size [?]
* save to data/[nyt]/cooccur/[nyt_2006.bin]

4. Learn embedding
python train.py
--cooccur_folder [?]
--vocab_folder [?]
--data_type [nyt/ice/eu]
--embed_folder [?]
--emb_dim 50
--epoch 40

* temporal embedding: ewe_temporal.c [ref: glove_region.c]
* spatial embedding: ewe_spatial.c [ref: glove_region_multi_penalty]
* two set of embeddings for condition-independent word embedding and deviation embedding
* only one set of condition embedding

5.1 post-process enriched embedding
python test_[nyt/ice/eu]_emb.py
* save embedding to data/nyt/embed/enriched_[2006].txt
* save embedding to data/ice/embed/enriched_[usa].txt

5.2 test enriched embedding
python test_nyt_emb.py 
--dataset [1/2] 
--test 
--start_ind 0
--end_ind 20
--use_cxt_vector
--remove_mean
--use_cond_word_vocab
* save predictions to data/nyt/eval/result/

python test_ice_emb.py
--test
--start_ind 0
--end_ind 100
--use_cxt_vector[?]
--remove_mean[?]
--use_cond_word_vocab[?]
* save predictions to data/ice/eval/result

python test_eu_emb.py
--test
--use_cxt_vector[?]
--remove_mean[?]
--use_cond_word_vocab[?]


6. evaluate embedding on alignment tasks
* nyt
python eval_emb.py 
--test_fn ../data/nyt/eval/Testset/testset_[1/2].csv
--res_folder ../data/nyt/eval/result/

* ice
python eval_emb.py 
--test_fn ../data/ice/eval/ice_equivalents.txt
--res_folder ../data/ice/eval/result/

