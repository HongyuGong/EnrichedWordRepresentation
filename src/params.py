"""
params.py
"""
import os

src_folder = os.path.abspath("src/")
print("src_folder: {}".format(src_folder))
src_model_folder = os.path.join(src_folder, "model")
data_folder = os.path.abspath("../data/")
raw_prefix = "raw_data"
cond_prefix = "cond_data"
vocab_prefix = "vocab"
cooccur_prefix = "cooccur"
embed_prefix = "embed"
sep_emb_prefix="word_or_cond_vectors"
dev_emb_prefix="word_and_cond_vectors"
sep_emb_cxt_prefix="word_or_cond_cxt_vectors"
dev_emb_cxt_prefix="word_and_cond_cxt_vectors"
eval_prefix = "eval"
res_prefix = "result"


# ---------------- nyt data (temporal) ----------------
nyt_folder = os.path.join(data_folder, "nyt")
nyt_corpus = "articles-search-1990-2016.json"
nyt_raw_data_folder = os.path.join(nyt_folder, raw_prefix)
nyt_cond_data_folder = os.path.join(nyt_folder, cond_prefix)
nyt_vocab_data_folder = os.path.join(nyt_folder, vocab_prefix)
nyt_cooccur_folder = os.path.join(nyt_folder, cooccur_prefix)
nyt_embed_folder = os.path.join(nyt_folder, embed_prefix)
nyt_eval_data_folder = os.path.join(nyt_folder, eval_prefix)
nyt_eval_res_folder = os.path.join(nyt_eval_data_folder, res_prefix)
time_list = range(1990, 2017)
# frequentcy threshold 
nyt_word_ft = 1
global_nyt_word_ft = 100

# ------------------ ice data (spatial) ------------------                                   
ice_folder = os.path.join(data_folder, "ice")
ice_raw_data_folder = os.path.join(ice_folder, raw_prefix)
ice_cond_data_folder = os.path.join(ice_folder, cond_prefix)
ice_vocab_data_folder = os.path.join(ice_folder, vocab_prefix)
ice_cooccur_folder = os.path.join(ice_folder, cooccur_prefix)
ice_embed_folder = os.path.join(ice_folder, embed_prefix)
ice_eval_data_folder = os.path.join(ice_folder, eval_prefix)
ice_eval_res_folder = os.path.join(ice_eval_data_folder, res_prefix)
region_list = ["canada", "east_africa", "hk", "india", "ireland", \
               "jamaica", "philippines", "singapore", "usa"]
ice_word_ft = 1
global_ice_word_ft = 5

#-------------- temporary for euphemism--------------------
eu_folder = os.path.join(data_folder, "eu")
eu_raw_data_folder = os.path.join(eu_folder, raw_prefix)
eu_cond_data_folder = os.path.join(eu_folder, cond_prefix)
eu_vocab_data_folder = os.path.join(eu_folder, vocab_prefix)
eu_cooccur_folder = os.path.join(eu_folder, cooccur_prefix)
eu_embed_folder = os.path.join(eu_folder, embed_prefix)
eu_eval_data_folder = os.path.join(eu_folder, eval_prefix)
eu_eval_res_folder = os.path.join(eu_eval_data_folder, res_prefix)
domain_list = ["wiki", "reddit"]
eu_word_ft = 20
global_eu_word_ft = 1



