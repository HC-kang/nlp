import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/nlp')

# # 희소 표현
# from google.colab import files
# files_uploaded=files.upload()

import pandas as pd
import numpy as np

class2 = pd.read_csv('class2.csv')


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

train_x = label_encoder.fit_transform(class2['class2'])
train_x

# 횟수 기반 임베딩
# Counter Vector    

# 출현 빈도 사용, 인코딩 >> 벡터화
# 토큰 >> 출현빈도 확인 >> 인코딩 >> 벡터로 변환

from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is last chance.',
    'and if you do not have this chance.',
    'you will never get any chance.',
    'will you do get this one?',
    'please, get this chance',
]

vect = CountVectorizer()
vect.fit(corpus)
vect.vocabulary_
len(vect.vocabulary_)

vect.transform(['you will never get any chance']).toarray()

# 불용어 (stopwords)
stopwords = ['and', 'is', 'this', 'will', 'please']
vect = CountVectorizer(stop_words=stopwords).fit(corpus)
vect.vocabulary_
len(vect.vocabulary_)

# TF-IDF
# Term Frequency, Inverse Document Frequency

# 특정 문서 내에서 단어의 출현 빈도가 높거나, 전체 문서에서 특정 단어가 포함된 문서가 적을수록 TF-IDF가 높다.

from sklearn.feature_extraction.text import TfidfVectorizer
doc = ['I love machine learning', 'I love deep learning', 'I like co-work']
tfidf_vect = TfidfVectorizer(min_df = 1)
tfidf_matrix = tfidf_vect.fit_transform(doc)
doc_distance = (tfidf_matrix * tfidf_matrix.T)
print(doc_distance.toarray())
doc_distance.shape

# word2vec
import nltk
nltk.download('popular')

# from google.colab import files
# file_uploaded = files.upload()

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import gensim
from gensim.models import Word2Vec

sample = open('peter.txt', 'r', encoding='UTF-8')
s = sample.read()
s

f = s.replace('\n', ' ')
f
data = [ ]
for i in sent_tokenize(f):
    temp = [ ]
    for j in word_tokenize(i):
        temp.append(j.lower())
    data.append(temp)

print(data)

# word2vec
# cbow
# 주변 단어를 통해서 중심 단어 예측

model1 = Word2Vec(data, min_count = 1,
                                vector_size = 100, window = 5, sg = 0)

# sg = 0 >> cbow, sg = 1 >> skip_gram

print(model1.wv.similarity('peter', 'wendy'))
print(model1.wv.similarity('peter', 'hook'))

model2 = gensim.models.Word2Vec(data, min_count = 1,
                                vector_size = 100, window = 5, sg = 1)

# sg = 0 >> cbow, sg = 1 >> skip_gram

print(model2.wv.similarity('peter', 'wendy'))
print(model2.wv.similarity('peter', 'hook'))


from gensim.models import FastText


model = FastText(corpus, vector_size = 4, window = 3, min_count = 1)

model.wv.similarity('peter', 'wendy')
model.wv.similarity('peter', 'hook')

