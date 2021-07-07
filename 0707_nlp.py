# !git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
# ls
# cd Mecab-ko-for-Google-Colab/
# !bash install_mecab-ko_on_colab190912.sh
# !pip install konlpy

from konlpy.tag import Mecab
mecab=Mecab()

import os 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

path_to_file = '/Users/heechankang/projects/pythonworkspace/git_study/nlp/korean-english-park.train.ko'

with open(path_to_file, 'r', encoding = 'UTF-8') as f:
  raw = f.read().splitlines()

  print('Data Size:', len(raw))
  print('Example:')
  for sen in raw[0:100][::20]: print('>>', sen)

print(len(raw[3]))
print(type(raw))

min_len = 999
max_len = 0
sum_len = 0

for sen in raw:
  length = len(sen)
  if min_len > length: min_len = length
  if max_len < length: max_len = length
  sum_len += length

print('문장의 최단 길이:', min_len)
print('문장의 최장 길이:', max_len)
print('문장의 평균 길이:',sum_len//len(raw))

sentence_length = np.zeros((max_len), dtype = np.int)
for sen in raw:
  sentence_length[len(sen)-1] +=1

plt.bar(range(max_len), sentence_length, width = 1.0)
plt.title('Sentence Length Distribution')

def check_sentence_with_length(raw, length):
    count = 0
    for sen in raw:
        if len(sen) == length:
            print(sen)
            count += 1
            if count > 100:
                return

check_sentence_with_length(raw, 1)

for idx, _sum in enumerate(sentence_length):
    if _sum >1500:
        print("Outlier index:", idx+1)

check_sentence_with_length(raw, 11)

min_len = 999
max_len = 0
sum_len = 0

cleaned_corpus = list(set(raw))
print('Data Size:', len(cleaned_corpus))

for sen in cleaned_corpus:
    length = len(sen)
    if min_len > length: min_len = length
    if max_len < length: max_len = length
    sum_len += length

print('문장의 최단 길이:', min_len)
print('문장의 최장 길이:', max_len)
print('문장의 평균 길이:', sum_len//len(cleaned_corpus))

check_sentence_with_length(cleaned_corpus, 11)

max_len = 150
min_len = 0

filtered_corpus = [s for s in cleaned_corpus if (len(s) < max_len) & (len(s) >= min_len)]

sentence_length = np.zeros((max_len), dtype = np.int)

for sen in filtered_corpus:
    sentence_length[len(sen)-1] +=1

plt.bar(range(max_len), sentence_length, width = 1.0)
plt.title('Sentence Length Distribution')

def tokenize(corpus):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '')
    tokenizer.fit_on_texts(corpus)

    tensor = Tokenizer.texts_to_sequences(corpus)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding = 'post')
    return tensor, tokenizer

# 정제된 데이터를 공백 기반으로 토큰화하여 저장하는 코드 작성
split_corpus = []

for kor in filtered_corpus:
  split_corpus.append(kor.split())


split_tensor, split_tokenizer = tokenize(split_corpus)
print("Split Vocab Size:", len(split_tokenizer.index_word))

split_tensor, split_tokenizer = tokenize(filtered_corpus)
print("Split Vocab Size:", len(split_tokenizer.index_word))

for idx, word in enumerate(split_tokenizer.word_index):
    if idx > 10: break
    print(idx, ':', word)
          
# 밝+혔다 // 밝+히다, 밝+다
# 위에서 사용한 코드를 활용해 MeCab 단어사전 만들기
# Hint : mecab.morphs() 를 사용해서 형태소 분석

def mecab_split(sentence):
    return mecab.morphs(sentence)

mecab_corpus = []

for kor in filtered_corpus:
    mecab_corpus.append(mecab_split(kor))


mecab_tensor, mecab_tokenizer = tokenize(mecab_corpus)
print('MeCab Vocab Size:', len(mecab_tokenizer.index_word))

# Case 1: tokenizer.sequence_to_texts()

#texts = mecab_tokenizer.sequences_to_texts(mecab_tensor[100])
texts = mecab_tokenizer.sequences_to_texts(mecab_tensor[100:101])
print(texts)


# Case 1: tokenizer.sequence_to_texts()

#texts = mecab_tokenizer.sequences_to_texts(mecab_tensor[100])
texts = mecab_tokenizer.sequences_to_texts([mecab_tensor[100]])
print(texts[0])

# Case 2: tokenizer.index_word
sentence = ''
for w in mecab_tensor[100]:
    if w == 0: continue
    sentence += mecab_tokenizer.index_word[w]

print(sentence)

mecab_tensor[100:101]
mecab_tensor[100]