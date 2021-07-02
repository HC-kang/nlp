import os
from __future__ import print_function
from gensim.models import KeyedVectors
import gensim
from gensim.models import Word2Vec


os.chdir('/content/drive/MyDrive/AI_data/wiki.ko')
model_kr = KeyedVectors.load_word2vec_format('wiki.ko.vec')

find_similar_to = '열정'
# similarity 관련 함수들이 코랩에서만 작동함..
for similar_word in model_kr.similar_by_word(find_similar_to):
  print('word:{0}, similar_words:{1:.2f}'.format(similar_word[0], similar_word[1]))

model_kr.wv.most_similar(positive=['동물', '육식동물'], negative=['사람'])
print(similarities)

####################################################################################

# 횟수 / 예측 기반 임베딩
# GloVe
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook
plt.style.use('ggplot')

from sklearn.decomposition import PCA # 주성분 분석, 주요인 분석
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scriptes.glove2word2vec import glove2word2vec

glove_file = datapath('/content/drive/MyDrive/AI_data/0702_nlp_data/glove.6B.100d.txt')

word2vec_glove_file = get_tmpfile('/content/drive/MyDrive/AI_data/0702_nlp_data/glove.6B.100d_temp.txt')

glove2word2vec(glove_file, word2vec_glove_file)

model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
model.most_similar('bill')

model.most_similar('cherry')

model.most_similar(negative=['cherry'])

result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print('{}:{:.4f}'.format(*result[0]))

def analogy(x1, x2, y1):
    result = model.most_similar(positive = [y1, x2], negative=[x1])
    return result[0][0]

analogy('austrailia', 'beer', 'france')

analogy('tall', 'tallest', 'long')

print(model.doesnt_match('breakfast cereal dinner lunch brunch').split())


# Transformer Attention
# Seq2Seq
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import io
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

def unicode_to_ascii(s):
    return ' '.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c)!='Mn')

def preprocess_sentence(w):
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = '<start> ' + w + ' <end>'
    return w

en_sentence = u'May I borrow this book?'
sp_sentence = u'¿Puedo tomar prestado este libro?'

print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8'))

def create_dataset(path, num_examples):
    lines = io.open(path, encoding = 'UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)

def max_length(tensor):
  return max(len(t) for t in tensor)

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.tokenizer(filters = '')
    lang_tokenizer = fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(tensor, padding = 'post')
    
    return tensor, lang_tokenizer 

def load_dataset(path, num_tokenizer = None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang.tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

from google.colab import files
uploaded_file = files.upload()

num_examples = 30000
input_tensor, target_tensor, inp_lang, target_lang = load_dataset('spa', num_example)
