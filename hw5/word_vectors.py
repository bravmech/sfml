
import numpy as np
import pandas as pd
import os
import logging

from tqdm import tqdm

from gensim.models import KeyedVectors

from texts2counters import Texts2Counters, logged, FOLDER_PATH

MODEL_PATH = '/home/mtomilov/Downloads/sfml/news_upos_cbow_600_2_2018.vec.gz'

logging.basicConfig(level=logging.INFO)


class WordVectors:

    def __init__(self, which, model):
        self.which = which
        self.fname = FOLDER_PATH.format(self.which, 'npy')
        self.df = None
        self.model = model
        self.counters = None
        self.vectors = None

    @logged
    def load(self, limit=None, calc=False):
        if not self.df:
            logging.info('data frame loading...')
            self.df = pd.read_csv(FOLDER_PATH.format(self.which, 'csv'), sep='\t', encoding='utf8', nrows=limit)

        if os.path.exists(self.fname) and not calc:
            self.npy = np.load(self.fname)
            return

        self.counters = Texts2Counters(self.which, self.df)
        self.counters.load(calc)
        self.calc_vectors()
        self.save()

    @logged
    def save(self):
        np.save(self.fname, self.vectors)

    @logged
    def calc_vectors(self):
        self.vectors = np.zeros((len(self.counters.data), self.model.vector_size), dtype='float32')
        i = 0
        for item in tqdm(self.counters.data):
            self.vectors[i] = bag_to_vec(item['counter'], self.model)
            i += 1


def bag_to_vec(dct, model, size=None, freq=True):
    if size is None:
        size = model.vector_size
    text_vec = np.zeros((size,), dtype="float32")
    n_words = 0

    index2word_set = set(model.index2word)
    for word in dct:
        if word in index2word_set:
            count = dct[word] if freq else 1
            n_words = n_words + count
            text_vec = np.add(text_vec, model[word] * count)

    if n_words:
        text_vec /= n_words
    return text_vec


if __name__ == '__main__':
    try:
        model_w2v
    except NameError:
        logging.info('word2vec model loading...')
        model_w2v = KeyedVectors.load_word2vec_format(MODEL_PATH)

    # wv_train = WordVectors('train', model_w2v)
    wv_test = WordVectors('test', model_w2v)
    wv_test.load(None, True)
