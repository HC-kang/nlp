import pandas as pd
import urllib

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()] # list() 로 만들기

        data = data[1:]
    return data

train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

len(train_data)
len(test_data)

len(train_data[0])

from konlpy.tag import Okt

okt = Okt()
okt.pos(u'이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요')
okt.pos('이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요')

import json # python 
import os
from pprint import pprint

def tokenize(doc):
    return ['/'.join(t) for t in okt.pos(doc, norm = True)]

if os.path.isfile('train_docs.json'):
    with open('train_docs.json') as f:
        train_docs = json.load(f)
    with open('test_docs.json') as f:
        test_docs = json.load(f)

else:
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

    with open('train_docs.json', 'w', encoding ='utf-8') as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent = '\t')
    with open('test_docs.json', 'w', encoding = 'utf-8') as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent = '\t')

pprint(train_docs[0])


tokens = [t for d in train_docs for t in d[0]]

len(tokens)

import nltk

text = nltk.Text(tokens, name = 'NMSC')

print(len(text.tokens)) # 전체 토큰 개수

print(len(set(text.tokens))) # 중복을 제외한 토큰 수

pprint(text.vocab().most_common())

selected_words = [f[0] for f in text.vocab().most_common(100)]

def term_frequency(doc):
    return [doc.count(word) for word in selected_words]

train_x = [term_frequency(d) for d, _ in train_docs]
test_x = [term_frequency(d) for d, _ in test_docs]

train_y = [c for _, c in train_docs]
test_y = [c for _, c in test_docs]