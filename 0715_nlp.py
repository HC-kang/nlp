!pip install konlpy

!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git

cd Mecab-ko-for-Google-Colab/

!bash install_mecab-ko_on_colab190912.sh

import tensorflow as tf
import numpy as np

from konlpy.tag import Mecab
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import time
import os
import re
import io

!sudo apt -qq -y install fonts-nanum

# Commented out IPython magic to ensure Python compatibility.
import matplotlib as mpl
import matplotlib.pyplot as plt

# %config InlineBackend.figure_format = 'retina'

import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic')
mpl.font_manager._rebuild()

path_to_file1 = '/content/drive/MyDrive/AI_data/korean-english-news-v1/korean-english-park.train.ko'
path_to_file2 = '/content/drive/MyDrive/AI_data/korean-english-news-v1/korean-english-park.train.en'

with open(path_to_file1, 'r') as f:
    train_raw = f.read().splitlines()

print('Train Data Size:', len(train_raw))
print('Train_raw:', train_raw[:3])

with open(path_to_file2, 'r') as f:
    target_raw = f.read().splitlines()

print('target Data Size:', len(target_raw))
print('target_raw:', target_raw[:3])

"""### 한국어와 영어 병렬로 정렬하고 중복제거"""

cleaned_corpus = set(zip(train_raw, target_raw))
len(cleaned_corpus)

"""###한국어와 영어 개별로 중복제거"""

q, r = len(set(train_raw)), len(set(target_raw))
print(q, r)

"""### 중복을 제거한 target 셋 크기에 맞춰서 train셋 중복 제거"""

train_dic = {}
for i, j in enumerate(train_raw):
    train_dic[i] = j

target_dic = {}
for i, j in enumerate(target_raw):
    target_dic[i] = j

target_unique_dic = {}
for i, j in target_dic.items():
    if j not in target_unique_dic.values():
        target_unique_dic[i] = j

train_unique_dic = {}
for i, j in train_dic.items():
    if i in target_unique_dic.keys():
        train_unique_dic[i] = j
print(len(train_unique_dic), len(target_unique_dic))

cleaned_eng_corpus = {}
cleaned_kor_corpus = {}
mecab = Mecab()

def preprocess_sentence(train_unique_dic, target_unique_dic):
    for idx, sentence in target_unique_dic.items():
        sentence = re.sub(r'([?,!.])', r' \1', sentence)
        sentence = re.sub(r'[" "]', ' ', sentence)
        sentence = re.sub(r'[^a-zA-Z0-9.!,]+', ' ', sentence)
        sentence = sentence.strip()
        sentence_list = sentence.split()
        if len(sentence_list) <= 48:
            sentence = '<start> ' + sentence
            sentence += ' <end>'
            sentence = sentence.split()
            cleaned_eng_corpus[idx] = sentence

    for idx, sentence in train_unique_dic.items():
        sentence = re.sub(r'([?,!.])', r' \1', sentence)
        sentence = re.sub(r'[" "]', ' ', sentence)
        sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9.!,]+', ' ', sentence)
        result = mecab.morphs(sentence)
        if len(result) <= 50:
            cleaned_kor_corpus[idx] = result
    return cleaned_eng_corpus, cleaned_kor_corpus

def preprocess_sentence(train_unique_dic, target_unique_dic):
  for idx, sentence in target_unique_dic.items():
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence = re.sub(r'[" "]', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z0-9.!,]+", " ", sentence)
    sentence = sentence.strip()
    sentence_list = sentence.split()
    if len(sentence_list) <= 48:
      sentence = '<start> ' + sentence
      sentence += ' <end>'
      sentence = sentence.split()
      cleaned_eng_corpus[idx] = sentence

  for idx, sentence in train_unique_dic.items():
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    sentence = re.sub(r'[" "]', " ", sentence)
    sentence = re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9.!,]+", " ", sentence)
    result = mecab.morphs(sentence)
    if len(result) <= 50:
      cleaned_kor_corpus[idx] = result
  return cleaned_eng_corpus, cleaned_kor_corpus

cleaned_eng_corpus, cleaned_kor_corpus = preprocess_sentence(train_unique_dic, target_unique_dic)

print(cleaned_eng_corpus[100], cleaned_kor_corpus[100])

print(len(cleaned_eng_corpus), len(cleaned_kor_corpus))

set_temp1 = set(cleaned_eng_corpus.keys())
set_temp2 = set(cleaned_kor_corpus.keys())
set_temp3 = set_temp2.intersection(set_temp1)
len(set_temp3)

train_list = []
target_list = []

for i, j in cleaned_eng_corpus.items():
    if i in set_temp3:
        target_list.append(j)

for i, j in cleaned_kor_corpus.items():
    if i in set_temp3:
        train_list.append(j)

print(len(train_list), len(target_list))

del q
del r
del train_dic
del target_dic
del train_unique_dic
del target_unique_dic
del cleaned_eng_corpus
del cleaned_kor_corpus
del set_temp1
del set_temp2
del set_temp3



maxlen = 50
def tokenize(corpus):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', num_words=20000)
    tokenizer.fit_on_texts(corpus)

    tensor = tokenizer.texts_to_sequences(corpus)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=maxlen)
    return tensor, tokenizer

enc_tensor, enc_vocab = tokenize(train_list)
dec_tensor, dec_vocab = tokenize(target_list)

print("korean vocab size :", len(enc_vocab.index_word))
print("English vocab size :", len(dec_vocab.index_word))

print(len(enc_tensor[12400]), len(dec_tensor[12400]))

"""## 모델 설계

### 바다나우 어텐션
"""

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.w_dec = tf.keras.layers.Dense(units)
        self.w_enc = tf.keras.layers.Dense(units)
        self.w_com = tf.keras.layers.Dense(1)

    def call(self, h_enc, h_dec):
        # h_enc == [batch x length x units]
        # h_dec == [batch x units]

        h_enc = self.w_enc(h_enc)
        h_dec = tf.expand_dims(h_dec, 1)
        h_dec = self.w_dec(h_dec)

        score = self.w_com(tf.nn.tanh(h_dec + h_enc))

        attn = tf.nn.softmax(score, axis = 1)

        context_vec = attn * h_enc
        context_vec = tf.reduce_sum(context_vec, axis = 1)

        return context_vec, attn

"""### Encoder / Decoder"""

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences = True)
    
    def call(self, x):
        out = self.embedding(x)
        out = self.gru(out)

        return out

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units,
                                       return_sequences = True,
                                       return_state = True)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, h_dec, enc_out):
        context_vec, attn = self.attention(enc_out, h_dec)

        out = self.embedding(x)
        out = tf.concat([tf.expand_dims(context_vec, 1), out], axis=-1)

        out, h_dec = self.gru(out)
        out = tf.reshape(out, (-1, out.shape[2]))
        out = self.fc(out)

        return out, h_dec, attn

batch_size = 48
src_vocab_size = len(enc_vocab.index_word) + 1
tgt_vocab_size = len(dec_vocab.index_word) + 1

units = 128
embedding_dim =128

encoder = Encoder(src_vocab_size, embedding_dim, units)
decoder = Decoder(tgt_vocab_size, embedding_dim, units)

sequence_len = 50

sample_enc = tf.random.uniform((batch_size, sequence_len))
sample_out = encoder(sample_enc)
print('Encoder output :', sample_out.shape)

sample_state = tf.random.uniform((batch_size, units))
sample_logits, h_dec, attn = decoder(tf.random.uniform((batch_size, 1)),
                                     sample_state, sample_out)
print('Decoder output :', sample_logits.shape)
print('Decoder Hidden state :',h_dec.shape)
print('Attention :', attn.shape)

"""Loss설계

"""

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    return tf.reduce_mean(loss)

"""## 훈련1 train step"""

@tf.function
def train_step(src, tgt, encoder, decoder, optimizer, dec_tok):
    bsz = src.shape[0]
    loss =0

    with tf.GradientTape() as tape:
        enc_out = encoder(src)
        h_dec = enc_out[:, -1]

        dec_src = tf.expand_dims([dec_tok.word_index['<start>']]*bsz, 1)

        for t in range(1, tgt.shape[1]):
            pred, h_dec, _ = decoder(dec_src, h_dec, enc_out)

            loss += loss_function(tgt[:, t], pred)
            dec_src = tf.expand_dims(tgt[:, t], 1)

    batch_loss = (loss/int(tgt.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

!pip install tqdm

from tqdm.notebook import tqdm
import random
epochs = 5

for epoch in range(epochs):
    total_loss = 0
    idx_list = list(range(0, enc_tensor.shape[0], batch_size))
    random.shuffle(idx_list)
    t = tqdm(idx_list)

    for (batch, idx) in enumerate(t):
        batch_loss = train_step(enc_tensor[idx:idx+batch_size],
                                dec_tensor[idx:idx+batch_size],
                                encoder,
                                decoder,
                                optimizer,
                                dec_vocab)
        total_loss += batch_loss

        t.set_description_str('Epoch %2d' %(epoch+1))
        t.set_postfix_str('Loss %.4f'%(total_loss.numpy()/(batch+1)))

def preprocess_sentence(sentence):
    sentence = re.sub(r'([?,!.])', r' \1', sentence)
    sentence = re.sub(r'[" "]', ' ', sentence)
    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9.!,]+', ' ', sentence)
    result = mecab.morphs(sentence)
    return result

def evaluate(sentence, encoder, decoder):
    attention = np.zeros((dec_tensor.shape[-1], enc_tensor.shape[-1]))

    sentence = preprocess_sentence(sentence)
    inputs = enc_vocab.texts_to_sequences([sentence])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                           maxlen=enc_tensor.shape[-1],
                                                           padding = 'post')
    result = ''
    enc_out = encoder(inputs)

    dec_hidden = enc_out[:, -1]
    dec_input = tf.expand_dims([dec_vocab.word_index['<start>']],0)

    for t in range(dec_tensor.shape[-1]):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention[t] = attention_weights.numpy()

        prediction_id = \
        tf.argmax(tf.math.softmax(predictions, axis = -1)[0]).numpy()

        result += dec_vocab.index_word[prediction_id] + ' '

        if dec_vocab.index_word[prediction_id] == '<end>':
            return result, sentence, attention_weights

        dec_input = tf.expand_dims([prediction_id], 0)
    return result, sentence, attention

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(1,1,1)
    ax.matshow(attention, cmap = 'viridis')

    fontdict = {'fontsize' : 14}
    ax.set_xtickerlabels(['']+sentence, fontdict = fontdict, rotation = 90)
    ax.set_ytickerlabels(['']+predicted_sentence, fontdict = fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()

def translate(sentence, encoder, decoder):
    result, sentence, attention = evaluate(sentence, encoder, decoder)
    print('Input: %s' %(sentence))
    print('Predicted translation: {}'.format(result))
    
    attention = attention[:len(result), :len(sentence)]
    plot_attention(attention, sentence, result.split(' '))

"""## 번역하기"""

translate('일곱 명의 사망자가 발생했다.', encoder, decoder)





