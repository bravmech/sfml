
import bs4
import collections as co
import json
import math
import pickle
import pymorphy2
import re
import logging
import pandas as pd
import os
from functools import wraps

from tqdm import tqdm

FOLDER_PATH = '/home/mtomilov/Downloads/sfml/week5/all/{}.{}'

logging.basicConfig(level=logging.INFO)


def logged(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        if args:
            logging.info('{}.{} was called...'.format(args[0].__class__.__name__, func.__name__))
        else:
            logging.info('{} was called...'.format(func.__name__))
        return func(*args, **kwargs)
    return with_logging


class Texts2Counters:

    POS_MAP = {'ADJF': '_ADJ', 'NOUN': '_NOUN', 'VERB': '_VERB'}
    RUSSIAN_WORD_RE = "([А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)*)"

    def __init__(self, which, df):
        self.which = which
        self.fname = FOLDER_PATH.format(self.which, 'pickle')
        self.df = df
        self.data = []
        self.morph = pymorphy2.MorphAnalyzer()

    @logged
    def load(self, calc=False):
        if os.path.exists(self.fname) and not calc:
            with open(self.fname, 'rb') as f:
                self.data = pickle.load(f)
            return

        for i in range(self.df.shape[0]):
            self.data.append({
                'title': self.df.loc[i, 'name'],
                'text': self.df.loc[i, 'description']
            })
        self.calc_counters(need_pos=True)
        self.save()

    @logged
    def save(self):
        with open(self.fname, 'wb') as f:
            pickle.dump(self.data, f)

    def get_counter(self, text, need_pos=False) -> dict:
        words = re.findall(self.RUSSIAN_WORD_RE, text, re.U)
        reswords = []

        for w in words:
            wordform = self.morph.parse(w)[0]
            try:
                if wordform.tag.POS in ['ADJF', 'NOUN', 'VERB']:
                    if need_pos:
                        reswords.append(wordform.normal_form + self.POS_MAP[wordform.tag.POS])
                    else:
                        reswords.append(wordform.normal_form)
            except AttributeError:
                pass

        stat = co.Counter(reswords)
        # stat = {a: stat[a] for a in stat.keys() if stat[a] > 1}
        return dict(stat.most_common())

    @logged
    def calc_counters(self, need_pos=False):
        for item in tqdm(self.data):
            item['counter'] = self.get_counter(item['text'], need_pos)


def cosine_similarity(a, b):
    if not a or not b:
        return 0
    sumab = sum(a[na] * b[na] for na in a if na in b)
    suma2 = sum(a[na] * a[na] for na in a)
    sumb2 = sum(b[nb] * b[nb] for nb in b)
    return sumab / math.sqrt(suma2 * sumb2)


if __name__ == '__main__':
    pass
