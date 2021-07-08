
# Bag of Word(BoW) : 단어의 등장순서를 고려하지 않는 빈도수 기반의 단어 표현 방법
######
doc1 = 'John likes to watch movies. Mary likes movies too.'

BoW1 = {'John':1, 'likes':2, 'to':1, 'watch':1, 'movies':2, 'Mary':1, 'too':1}

# !pip install konlpy

from konlpy.tag import Okt
import re
okt = Okt()

# token = re.sub('(\.)', '', '정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.')
token = re.sub('(\.)', '', '소비자는 주로 소비하는 상품을 기준으로 물가상승률을 느낀다.')
# 정규표현식을 활용해 온점을 제거

token = okt.morphs(token)
# okt 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에 token에 넣어주기

word2index = {}
bow = []
for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)
        # token을 읽으면서, word2index에 없는 단어는 새로 추가하고, 이미 있는 단어는 pass
        bow.insert(len(word2index)-1,1)
        # bow 전체에 전부 기본값 1 넣어주기. 단어 개수는 최소 1개이기 때문.

    else:
        index = word2index.get(voca)
        # 재등장하는 단어의 인덱스를 받아오기
        bow[index] = bow[index]+1
        # 재등장하는 단어는 해당 인덱스에 1 더해주기
print(word2index)

bow

## Keras Tokenizer 를 활용한 Bow
##########

from tensorflow.keras.preprocessing.text import Tokenizer

sentence = ['John likes to watch movies. Mary likes movies too! Mary also likes to watch football games.']

def print_bow(sentence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentence) # 단어장 생성
    bow = dict(tokenizer.word_counts) # 각 단어와 그 빈도를 bow에 저장

    print('Bag of words: ', bow)
    print('단어장(vocab)의 크기:', len(tokenizer.word_counts)) # 중복을 제거한 단어들의 개수

print_bow(sentence)


## Scikit-learn CountVectorizer를 활용한 BoW
######
from sklearn.feature_extraction.text import CountVectorizer

sentence = ['John likes to watch movies. Mary likes movies too! Mary also likes to watch football games.']

vector = CountVectorizer()
print('Bag of Words:', vector.fit_transform(sentence).toarray()) # 코퍼스로부터 각 단어의 빈도수를 기록
print('각 단어의 인덱스:', vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되는지 보여줌


## 불용어를 제거한 BoW만들기
#####
# 사용자 정의 불용어 사용
from sklearn.feature_extraction.text import CountVectorizer

text = ["Family is not an important thing. It's everything"]
vect = CountVectorizer(stop_words=['the', 'a', 'an', 'is', 'not'])

print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)

# CountVectorizer에서 제공하는 자체 불용어
from sklearn.feature_extraction.text import CountVectorizer

text = ["Family is not an important thing. It's everything"]
vect = CountVectorizer(stop_words='english')
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)

# nltk에서 지원하는 불용어 사용
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
nltk.download('stopwords')

text = ["Family is not an important thing. It's everything"]
sw = stopwords.words('english')
vect = CountVectorizer(stop_words=sw)

print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)

## DTW(Document-Term Matrix)
# 다수의 문서에서 등장하는 각 단어들의 빈도를 행렬로 표현 한 것
# 다수의 문서에 대해서 bow 를 하나의 행렬로 표현하고 부르는 용어
# 문서 1: I like Dog
# 문서 2: I like cat
# 문서 3: I like cat I like cat
import pandas as pd
content = [[0,1,1,1], [1,0,1,1],[2,0,2,2]]
df = pd.DataFrame(content)
df.index=['I like Dog', 'I like Cat', 'I like Cat I like Cat']
df.columns = ['cat', 'dog', 'I', 'like']
df

import numpy as np
from numpy import dot
from numpy.linalg import norm

doc1 = np.array([0,1,1,1])
doc2 = np.array([1,0,1,1])
doc3 = np.array([2,0,2,2])

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

print(cos_sim(doc1, doc2))
print(cos_sim(doc1, doc3))
print(cos_sim(doc2, doc3)) # 코사인 유사도는 0~1까지의 값을 가지며, 1에 가까울수록 유사도가 높다고 판단.


## Scikit-learn CountVectorizer 를 활용한 DTM 구현
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
          'John likes to watch movies',
          'Mary likes movies too',
          'Mary also likes to watch football games'
]

vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray())
print(vector.vocabulary_)



## 한계점
# > 1. 희소표현(sparse representation)
# > 2. 단순 빈도 수 기반 접근

## 극복? -> TF-IDF (Term Frequency-Inverse Document Frequency)
# > 자주 나오는 단어가, 다른 문서에서도 흔하게 나온다면 중요하지 않은 단어일 가능성이 높다.\
# \
# 모든 문서에서 자주 등장하는 단어는 중요도가 낮다고 판단하고, 특정 문서에서만 자주 등장하는 단어는 중요도가 높다고 판단한다.

from math import log
import pandas as pd

docs = [
        'John likes to watch movies and Mary likes movies too',
        'James likes to watch TV',
        'Mary also likes to watch football games',
]

vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

print('단어장의 크기:', len(vocab))
print(vocab)

N = len(docs)
N

def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df+1))+1

def tfidf(t, d):
    return tf(t,d) * idf(t)

result = []

for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]

        result[-1].append(tf(t,d))

tf_ = pd.DataFrame(result, columns = vocab)
tf_

result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=vocab, columns = ['IDF'])
idf_


result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(vocab)):
        t = vocab[j]

        result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)
tfidf_

'''
'0 : John likes to watch movies and Mary likes movies too',
'1 :James likes to watch TV',
'2 : Mary also likes to watch football games',
'''


from sklearn.feature_extraction.text import CountVectorizer

corpus = [
          'you know I want your love',
          'I like you',
          'what should I do'
]

vector = CountVectorizer()