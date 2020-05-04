"""
enriched_word_emb.py
 - EWE: Enriched Word Embedding
 - ref: dynamic_vocab.py
"""

import os
import numpy as np
from gensim.models import KeyedVectors
import copy


def cosSim(a, b):
    norm_a = np.linalg.norm(a) #, 2)
    norm_b = np.linalg.norm(b) #, 2)
    if (norm_a * norm_b == 0):
        return 0
    return np.dot(a, b) / float(norm_a * norm_b)


class EnrichedWordEmb():
    
    def __init__(self, vocab_fn, cond_fn, sep_emb_fn, dev_emb_fn,
                 sep_emb_cxt_fn=None, dev_emb_cxt_fn=None,
                 embed_folder=None, use_cxt_vector=False,
                 remove_mean=False, use_cond_word_vocab=False):
        self.vocab_fn = vocab_fn
        self.cond_fn = cond_fn
        # separate embedding: basic word and condition vector
        self.sep_emb_fn = sep_emb_fn
        # deviation embedding: word-condition vector
        self.dev_emb_fn = dev_emb_fn
        # separate context embedding: context word and condition vector
        self.sep_emb_cxt_fn = sep_emb_cxt_fn
        # deviation context embedding
        self.dev_emb_cxt_fn = dev_emb_cxt_fn
        self.embed_folder = embed_folder
        self.use_cxt_vector = use_cxt_vector
        self.remove_mean = remove_mean
        self.use_cond_word_vocab = use_cond_word_vocab
        print("use context vector: {}, remove mean: {}, use conditional vocabulary: {}".format(
            self.use_cxt_vector, self.remove_mean, self.use_cond_word_vocab))

        # vocabulary
        self.idx2word, self.word2idx, self.word_size = self.readVocab(self.vocab_fn)
        self.idx2cond, self.cond2idx, self.cond_size = self.readVocab(self.cond_vocab_fn)

        # embedding
        self.word_embed, self.cond_embed = self.readSepEmbed(self.sep_emb_fn)
        if use_cxt_vector:
            self.word_embed_cxt, self.cond_embed_cxt = self.readSepEmbed(self.sep_emb_cxt_fn)


    def readVocab(self, vocab_fn):
        idx2word = []
        word2idx = {}
        with open(vocab_fn, "r") as fin:
            for line in fin:
                word, freq = line.strip().split()
                word2idx[word] = len(idx2word)
                idx2word.append(word)
        return idx2word, word2idx, len(idx2word)


    def _parse_line(self, line):
        seq = line.strip().split()
        word = seq[0]
        vec = np.array(seq[1:], "float")
        return word, vec

    
    def readSepEmbed(self, sep_emb_fn):
        word_embed = []
        cond_embed = []
        cond_list = []
        with open(sep_emb_fn, "r") as fin:
            # read word embedding
            word_cnt = 0
            for line in f:
                word, vec = self._parse_line(line)
                word_embed.append(copy.deepcopy(vec))
                word_count += 1
                if word_count >= self.word_size:
                    break
            # read condition embedding
            cond_cnt = 0
            for line in f:
                cond, vec = self._parse_line(line)
                cond_list.append(cond)
                cond_embed.append(copy.deepcopy(vec))
                cond_cnt += 1
                if cond_cnt >= self.cond_size:
                    break
            print("Sanity check - cond_list: {}".format(cond_list))
        return word_embed, cond_embed


    def synEmbed(self, cond, is_cxt_vector=False):
        """
        @func: synthesize enriched embedding from word, condition
        and deviation embedding
        """
        cond_idx = self.cond2idx[cond]
        if is_cxt_vector:
            word_embed, cond_embed = self.word_embed_cxt, self.cond_embed_cxt
            dev_emb_fn = self.dev_emb_cxt_fn
        else:
            word_embed, cond_embed = self.word_embed, self.cond_embed
            dev_emb_fn = self.dev_emb_fn
            
        dev_embed = []
        with open(dev_emb_fn, "r") as fin:
            for line in fin:
                if count >= cond_idx and (count-cond_idx) % self.cond_size == 0:
                    word, vec = self._parse_line(line)
                    dev_embed.append(copy.deepcopy(vec))
        assert len(dev_embed) == self.vocab_size

        cond_vec = cond_embed[cond_idx]
        enriched_embed = np.multiply(word_embed, cond_vec) + dev_embed
        return enriched_embed

    
    def genEnrichedEmbed(self, cond):
        """
        @func: generate enriched word embedding
        """
        enriched_embed = self.synEmbed(cond, is_cxt_vector=False)
        if self.use_cxt_vector:
            enriched_embed_cxt = self.synEmbed(cond, is_cxt_vector=True)
            enriched_embed = 0.5 * (enriched_embed + enriched_embed_cxt)
        if self.remove_mean:
            mean_vec = np.mean(enriced_embed, axis=0)
            enriched_embed = enriched_embed - mean_vec
        save_fn = os,path.join(self.embed_folder, "enriched_"+str(cond)+".txt")
        
        # Only use words occurring in the given condition
        if self.use_cond_word_vocab:
            print("Using conditional word vocabulary...")
            cond_word_vocab_fn = os.path.join(vocab_folder, str(cond)+".txt")
            _, cond_idx2word, _ = self.readVocab(cond_word_vocab_fn)

            cond_words = []
            cond_vecs = []
            for (word, vec) in zip(idx2word, enriched_embed):
                if word in cond_word2idx:
                    cond_words.append(word)
                    cond_vecs.append(copy.deepcopy(vec))
            self.saveEmbed(cond_words, cond_vecs, save_fn)
        else:
            print("Using whole word vocabulary...")
            self.saveEmbed(self.idx2word, enriched_embed, save_fn)


    def saveEmbed(self, words, vecs, save_fn):
        with open(save_fn, "w") as fout:
            fout.write(str(len(words)) + " " + str(len(vec[0])) + "\n")
            for (word, vec) in zip(words, vecs):
                fout.write(" ".join([word] + [str(val) for val in vec]) + "\n")
        

    def findIntraNeighbors(self, cond, words, topn=20):
        """
        @func: find word neighbors in terms of cosine similarity
                     in the same condition
        """
        embed_fn = os.path.join(self.embed_folder, "enriched_"+str(cond)+".txt")
        if not os.path.isfile(embed_fn):
            self.genEnrichedEmbed(cond)
        word_vectors = KeyedVectors.load_word2vec_format(embed_fn, binary=False)
        neb_dict = dict()
        for word in words:
            try:
                nebs = word_vectors.similar_by_word(word, topn=topn)
                neb_dict[word] = nebs[:]
            except:
                print("Not found in {}: {}".format(cond, word))
        return neb_dict


    def findInterNeighbors(self, src_words, src_cond, trg_cond, topn=20):
        src_embed_fn = os.path.join(self.embed_folder, "enriched_"+str(src_cond)+".txt")
        if not os.path.isfile(src_embed_fn):
            self.genEnrichedEmbed(src_cond)
        trg_embed_fn = os.path.join(self.embed_folder, "enriched_"+str(trg_cond)+".txt")
        if not os.path.isfile(trg_embed_fn):
            self.genEnrichedEmbed(trg_cond)
                
        src_embed = KeyedVectors.load_word2vec_format(src_embed_fn, binary=False)
        trg_embed = KeyedVectors.load_word2vec_format(trg_embed_fn, binary=False)

        neb_dict = {}
        for word in src_words:
            try:
                src_vec = src_embed[word]
            except:
                neb_dict[word] = []
                print("Not found in cond {}: {}".format(src_cond, word))
                continue
            nebs = trg_embed.similar_by_vector(src_vec, topn=topn)
            nebs = [(neb[0]+"-"+str(trg_cond), neb[1]) for neb in nebs]
            neb_dict[word] = nebs[:]
        return neb_dict


    def sortWordChanges(self, src_cond, trg_cond, topn=20):
        src_embed_fn = os.path.join(self.embed_folder, "enriched_"+str(src_cond)+".txt")
        if not os.path.isfile(src_embed_fn):
            self.genEnrichedEmbed(src_cond)
        trg_embed_fn = os.path.join(self.embed_folder, "enriched_"+str(trg_cond)+".txt")
        if not os.path.isfile(trg_embed_fn):
            self.genEnrichedEmbed(trg_cond)

        src_word_vectors = KeyedVectors.load_word2vec_format(src_embed_fn, binary=False)
        trg_word_vectors = KeyedVectors.load_word2vec_format(trg_embed_fn, binary=False)
        word_sim_dict = {}
        for word in self.idx2word:
            if word in src_word_vectors and word in trg_word_vectors:
                src_vec = src_word_vectors[word]
                trg_vec = trg_word_vectors[word]
                sim = cosSim(src_vec, trg_vec)
                word_sim_dict[word] = sim
            else:
                continue
        sorted_vocab = sorted(word_sim_dict.items(), key=lambda tup: tup[1], reverse=True)
        print("words between {} and {} with high stability:\n{}".format(
            src_cond, trg_cond, sorted_vocab[:topn]))
        print("words between {} and {} with low stability:\n{}".format(
            src_cond, trg_cond, sorted_vocab[-1*topn:]))
        
        

        
