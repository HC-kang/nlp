pip install customized_konlpy

from ckonlpy.tag import Twitter
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from nltk import FreqDist
from functools import reduce
import os
import re
import matplotlib.pyplot as plt

train_file = '/content/drive/MyDrive/AI_data/nlp_qa/qa1_single-supporting-fact_train_kor.txt'
test_file = '/content/drive/MyDrive/AI_data/nlp_qa/qa1_single-supporting-fact_test_kor.txt'

i = 0
lines = open(train_file, 'rb')
for line in lines:
    line = line.decode('utf-8').strip()
    i = i+1
    print(line)
    if i==20:
        break

def read_data(dir):
    stories, questions, answers = [], [], []
    story_temp = []
    lines = open(dir, 'rb')

    for line in lines:
        line = line.decode('utf-8') # b' 분리
        line = line.strip() # \n 제거
        idx, text = line.split(' ', 1) # 맨 앞에 있는 id number 분리

        if int(idx) == 1:
            story_temp = []

        if '\t' in text: # 현재 읽는 줄이 질문(tab) 답변(tab) 인경우
            question, answer, _ = text.split('\t')
            stories.append([x for x in story_temp if x])
            questions.append(question)
            answers.append(answer)
        else:
            story_temp.append(text)
    lines.close()
    return stories, questions, answers

train_data = read_data(train_file)
test_data = read_data(test_file)

train_stories, train_questions, train_answers = read_data(train_file)
test_stories, test_questions, test_answers = read_data(test_file)

print('훈련용 스토리의 개수:', len(train_stories))
print('훈련용 질문의 개수:', len(train_questions))
print('훈련용 답변의 개수:', len(train_answers))
print('테스트용 스토리의 개수:', len(test_stories))
print('테스트용 질문의 개수:', len(test_questions))
print('테스트용 답변의 개수:', len(test_answers))

train_stories[3678]

train_questions[3678]

train_answers[3678]

# 토큰화
def tokenize(sent):
    return [x.strip() for x in re.split('(\\+)?', sent) if x.strip()]

def preprocess_data(train_data, test_data):
    counter = FreqDist()

    # 두 문장의 story를 하나의 문장으로 통합하는 함수
    flatten = lambda data: reduce(lambda x, y: x+y, data)

    # 각 샘플의 길이를 저장하는 리스트
    story_len = []
    question_len = []

    for stories, questions, answers in [train_data, test_data]:
        for story in stories:
            stories = tokenize(flatten(story))
            story_len.append(len(stories))
            for word in stories: # 단어 집합에 단어 추가
                count[word] += 1
        for question in questions:
            question = tokenize(question)
            qestion_len.append(len(question))
            for word in question:
                counter[word]+=1
        for answer in answers:
            answer = tokenize(answer)
            for word in answer:
                counter[word] += 1

        # 단어 집합 생성
        word2idx = {word:(idx+1) for idx, (word, _) in enumerate(counter.most_common())}
        idx2word = {idx:word for word, idx in word2idx.items()}

        story_max_len = np.max(story_len)
        question_max_len = np.max(question_len)

        return word2idx, idx2word, story_max_len, question_max_len

word2idx, idx2word, story_max_len, question_max_len = preprocess_data(train_data, test_data)

print(word2idx)

vocab_size = len(word2idx) +1
print(vocab_size)

print('스토리의 최대 길이:', story_max_len)
print('질문의 최대 길이:', question_max_len)

def vectorize(data, word2idx, story_maxlen, question_maxlen):
    Xs, Xq, y = [], [], []
    flatten = lambda data: reduce(lambda x, y: x+ y, data)

    stories, questions, answers = data
    for story, question, answer in zip(stories, questions, answers):
        xs = [word2idx[w] for w in tokenize(flatten(story))]
        xq = [word2idx[w] for w in tokenize(question)]
        Xs.append(xs)
        Xq.append(xq)
        y.append(word2idx[answer])

    return pad_sequences(Xs, maxlen=story_maxlen),\
    pad_sequences(Xq, maxlen=question_maxlen),\
    to_categorical(y, num_classes=len(word2idx) + 1)

xstrain, xqtrain, ytrain = vectorize(train_data, word2idx, story_max_len, question_max_len)
xstest, xqtest, ytest = vectorize(test_data, word2idx, story_max_len, question_max_len)

print(xstrain.shape, xqtrain.shape, ytrain.shape, xstest.shape, xqtest.shape, ytest.shape)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Permute, dot, add, concatenate
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Activation

train_epochs = 120
batch_size = 32
embed_size = 50
lstm_size = 64
dropout_rate = 0.3

input_sequence = Input((story_max_len,))
question = Input((question_max_len,))

print('Stories:', input_sequence)
print('Question:', question)

# Embedding A
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_size,
                              output_dim = embed_size))
input_encoder_m.add(Dropout(dropout_rate))
# (샘플의 수, 문장의 최대 길이, 임베딩 벡터의 차원)

# Embedding C
# 임베딩 벡터의 차원을 질문의 최대 길이로
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_size,
                              output_dim = question_max_len))
input_encoder_c.add(Dropout(dropout_rate))
# (샘플의 수, 문장의 최대길이, 질문의 최대길이(임베딩 벡터의 차원)

# 질문을 위한 임베딩 Embedding B
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_size,
                               output_dim = embed_size,
                               input_length = question_max_len))
question_encoder.add(Dropout(dropout_rate))
# (샘플의 수, 질문의 최대길이, 임베딩 벡터의 차원)

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

print('input_encoded_m:', input_encoded_m)
print('input_encoded_c:', input_encoded_c)
print('question encoded', question_encoded)

match = dot([input_encoded_m, question_encoded], axes= -1, normalize=False) ##
match = Activation('softmax')(match)
print('Match shape:', match)

response = add([match, input_encoded_c])
response = Permute((2, 1))(response)
print('Response shape:', response)

answer = concatenate([response, question_encoded])
print('Answer shape:', answer)

answer = LSTM(lstm_size)(answer)
answer = Dropout(dropout_rate)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

model = Model([input_sequence, question], answer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit([xstrain, xqtrain],
                    ytrain, batch_size, train_epochs,
                    validation_data = ([xstest, xqtest], ytest))
model.save('model.h5')

plt.subplot(211)
plt.title('Accuracy')
plt.plot(history.history['acc'], color = 'g', label = 'train')
plt.plot(history.history['val_acc'], color = 'b', label = 'validation')
plt.legend(loc = 'best')

plt.subplot(212)
plt.title('Loss')
plt.plot(history.history['loss'], color = 'g', label = 'train')
plt.plot(history.history['val_loss'], color = 'b', label = 'validation')
plt.legend(loc = 'best')

plt.tight_layout()
plt.show()

# labels
ytest = np.argmax(ytest, axis = 1)

# get predictions
ytest_ = model.predict([xstest, xqtest])
ytest_ = np.argmax(ytest_, axis=1)

num_display = 30

print("{:18}|{:5}|{}".format("질문","실제값","예측값"))
print(39 * "_")

for i in range(num_display):
    question = " ".join([idx2word[x] for x in xqtest[i].tolist()])
    label = idx2word[ytest[i]]
    prediction = idx2word[ytest_[i]]
    print("{:20}|{:7}|{}".format(question, label, prediction))

import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    def get_angles(self, position, i, d_model):
        angles = 1/tf.pow(10000, (2*(i//2))/tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position= tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model = d_model
        )

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]   ### [1, :, :]

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

import matplotlib.pyplot as plt
import numpy as np

sample_pos_encoding = PositionalEncoding(50, 128)

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()


def scaled_dot_product_attention(query, key, value, mask):
    # query 크기 : (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
    # key 크기 : (batch_size, num_heads, key의 문장 길이, d_model/num_heads)
    # value 크기 : (batch_size, num_heads, value의 문잔 길이, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key의 문장길이)

    matmul_qk = tf.matmul(query, key, transpose_b = True)

    # 스케일링
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # 마스킹
    if mask is not None:
        logits += (mask * -1e9)

    # attention_weights : (batch_size, num_heads, query 문장길이, key의 문장 길이)
    attention_weights = tf.nn.softmax(logits, axis = -1)

    output = tf.matmul(attention_weights, value)

    return output, attention_weights

np.set_printoptions(suppress=True)
temp_k = tf.constant([[10, 0, 0],
                      [0, 10, 0],
                      [0, 0, 10],
                      [0, 0, 10]], dtype = tf.float32)
temp_v = tf.constant([[1, 0],
                      [10, 0],
                      [100, 5],
                      [1000, 6]], dtype = tf.float32)
temp_q = tf.constant([[0, 10, 0]], dtype = tf.float32)

temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)

print(temp_attn)
print(temp_out)

temp_q = tf.constant([[0, 0, 10]], dtype = tf.float32)
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn)
print(temp_out)

temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype = tf.float32) # 3,3
temp_out, temp_attn = scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
print(temp_attn)
print(temp_out)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name = 'multi_head_attention'):
        super(MultiHeadAttnetion, self).__init__(name = name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        # d_model을 num_head로 나눈 값 --> 64
        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units = d_model)
        self.key_dense = tf.keras.layers.Dense(units = d_model)
        self.value_dense = tf.keras.layers.Dense(units = d_model)

        # w0 dense layer
        self.dense = tf.kears.layers.Dense(units = d_model)

    # num_heads 개수만큼 q, k, v split 하는 함수
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape = (batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(inputs, perm = [0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. WQ, Wk, Wv에 해당하는 d_model 크기의 Dense layer를 지나게 한다.
        # q: (batch_size, query의 문장 길이, d_model)
        # k: (batch_size, key의 문장 길이, d_model)
        # v: (batch_size, value의 문장 길이, d_model)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 헤드 나누기
        # q: (batch_size, query의 문장 길이, d_model/num_heads)
        # k: (batch_size, key의 문장 길이, d_model/num_heads)
        # v: (batch_size, value의 문장 길이, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. 스케일드 닷 프로덕트 어텐션
        # (batch_size, num_heads, query의 문장 길이, d_model/num_heads)
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        # (batch_size, query의 문장 길이, num_heads,  d_model/num_heads)
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])
        
        # 4. 헤드 연결하기
        # (batch_size, query의 문장 길이, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      batch_size, -1, self.d_model)
        
        # 5. wo에 해당하는 dense layers 지니기
        outputs = self.dense(concat_attention)

        return outputs

