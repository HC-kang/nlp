from konlpy.tag import Okt
import re

okt = Okt()
token = re.sub('(\.)', '', '정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다')

token

okt.morphs(token)

token = okt.morphs(token)
token

word2idx = {} 
bow = [] # bag of words

for voca in token: # 토큰화 된 단어를 하나씩 불러오기
    if voca not in word2idx.keys(): # 단어사전에 voca가 없다면
        word2idx[voca] = len(word2idx) # 단어사전의 길이 len을 해당 voca의 value로 설정 - 순서 기입
        bow.insert(len(word2idx)-1, 1)
        # bow에 횟수를 1로 초기화
    else: # 만약 있으면
        index = word2idx.get(voca) # 해당 voca의 value값을 가져오기
        bow[index] = bow[index]+1 # value 값을 index로 bow에서 꺼내와서, 1 더하기.

print(word2idx)
print(bow)

# countervectorizer 이용
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know i want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray())
print(vector.vocabulary_)



################################################################################

from math import *
import pandas as pd

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]

voca = list(set(w for doc in docs for w in doc.split()))
voca
voca.sort()

print(voca)

len(docs)

N = len(docs)

def tf(t, d): # term frequency : 단어의 빈도
    return d.count(t)

def idf(t): # inversed document frequency : 역 문서 빈도
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df+1))

def tfidf(t, d):
    return tf(t,d) * idf(t)

# term frequency
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(voca)):
        t = voca[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = voca)
tf_

# idf
result = []
for j in range(len(voca)):
    t = voca[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index=voca, columns = ['IDF'])
idf_

# tf - idf
result = []
for i in range(N):
    result.append([])
    d = docs[i]
    for j in range(len(voca)):
        t = voca[j]

        result[-1].append(tfidf(t, d))

tfidf_ = pd.DataFrame(result, columns = voca)
tfidf_

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

texts = ['먹고 싶은 사과', '먹고 싶은 바나나', '길고 노란 바나나 바나나', '저는 과일이 좋아요']
token = Tokenizer()
token.fit_on_texts(texts)
print(token.word_index)

print(token.texts_to_matrix(texts, mode = 'count'))
print(token.texts_to_matrix(texts, mode = 'binary'))

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

'''
딥러닝 실행 순서
1. 데이터 수집 - 전처리
2. 모델링
3. 컴파일 : 환경설정 (train data)
4. 학습 (fit, fit_transform)
'''

import tensorflow as tf
tf.keras.layers.Dense(10)

from tensorflow.keras.layers import Dense
Dense(10)

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# 데이터 전처리
xs = np.array([-1., 0., 1., 2., 3., 4.], dtype = float)
ys = np.array([5., 6., 7., 8. ,9. ,10.], dtype = float)

# 모델링
model = Sequential()
model.add(Dense(1, input_dim = 1, activation = 'linear'))
model.compile(optimizer='sgd', loss = 'mse')

# 학습
model.fit(xs,ys, epochs = 1200, verbose = 0)

model.predict([10.0])

# 선형회귀 구현하기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
import numpy as np

X=[1,2,3,4,5,6,7,8,9] # 공부하는 시간
y=[11,22,33,44,53,66,77,87,95] # 각 공부하는 시간에 맵핑되는 성적

model = Sequential()
model.add(Dense(1, input_dim = 1, activation='linear'))
sgd = optimizers.SGD(lr=0.01)

model.compile(optimizer = sgd, loss = 'mse', metrics = 'accuracy')
model.fit(X, y, batch_size = 32, shuffle =False, epochs = 300)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(X, model.predict(X), 'b', X, y, 'k.')
ax.plot(X, y, 'r', X, y, 'k.')

import tensorflow as tf
import numpy as np

w = tf.Variable(2.)

def f(w):
    y = w**2
    z = 2*y + 5
    return z

with tf.GradientTape() as tape:
    z = f(w)

X=[1,2,3,4,5,6,7,8,9] # 공부하는 시간
y=[11,22,33,44,53,66,77,87,95] # 각 공부하는 시간에 맵핑되는 성적

W = tf.Variable(4.)
b = tf.Variable(1.)

@tf.function
def hypothesis(x):
    return W*x+b

x_test = [3.5, 5, 5.5, 6]

print(hypothesis(x_test).numpy())

@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred-y))

optimizer = tf.optimizers.SGD(lr=0.01)
for i in range(301):
    with tf.GradientTape() as tape:
        y_pred = hypothesis(X)
        cost = mse_loss(y_pred, y)

    gradients = tape.gradient(cost, [W, b])
    
    optimizer.apply_gradients(zip(gradients, [W,b]))

    if i%10 ==0:
        print('epoch : {:3} | W의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}'.format(i, W.numpy(), b.numpy(), cost))

x_test = [3.5, 5, 5.5, 6, 9.5]


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5, 5, 0.1)
y = sigmoid(x)

fig, ax = plt.subplots()
ax.plot(x, y, 'g')
ax.plot([0,0], [1.0, 0.0], ':')

plt.plot(x, y, 'g')
plt.plot([0,0], [1.0, 0.0], ':')
plt.title('sigmoid')

def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--') # W의 값이 0.5일때
plt.plot(x, y2, 'g') # W의 값이 1일때
plt.plot(x, y3, 'b', linestyle='--') # W의 값이 2일때
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
plt.plot(x, y2, 'g') # x + 1
plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = Sequential()
model.add(Dense(1, input_dim =1, activation = 'sigmoid'))
sgd = optimizers.SGD(lr = 0.01)

model.compile(optimizer = sgd, loss = 'binary_crossentropy')

model.fit(X, y, batch_size = 1, epochs = 300, shuffle = False)

print(model.predict([1, 2, 3, 4, 4.5]))
print(model.predict([11, 21, 31, 41, 500]))



import numpy as np
from tensorflow.keras.models import Sequential # 케라스의 Sequential()을 임포트
from tensorflow.keras.layers import Dense # 케라스의 Dense()를 임포트
from tensorflow.keras import optimizers # 케라스의 옵티마이저를 임포트

# 입력 벡터의 차원은 3입니다. 즉, input_dim은 3입니다.
X = np.array([[70,85,11],[71,89,18],[50,80,20],[99,20,10],[50,10,10]]) # 중간, 기말, 가산점
# 출력 벡터의 차원은 1입니다. 즉, output_dim은 1입니다.
y = np.array([73,82,72,57,34]) # 최종 성적

model = Sequential()
model.add(Dense(1, input_dim = 3, activation = 'linear'))
sgd = optimizers.SGD(lr = 0.01)

model.compile(optimizer = sgd, loss = 'binary_crossentropy')
model.fit(X, y, batch_size = 1, epochs = 1000, shuffle = False)



import numpy as np
from tensorflow.keras.models import Sequential # 케라스의 Sequential()을 임포트
from tensorflow.keras.layers import Dense # 케라스의 Dense()를 임포트
from tensorflow.keras import optimizers # 케라스의 옵티마이저를 임포트

# 입력 벡터의 차원은 2입니다. 즉, input_dim은 2입니다.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 출력 벡터의 차원은 1입니다. 즉, output_dim은 1입니다.
y = np.array([0, 1, 1, 1])

model = Sequential()
model.add(Dense(1, input_dim = 2, activation = 'sigmoid'))
model.compile(optimizer= 'sgd', loss = 'binary_crossentropy', metrics = 'binary_accuracy')
model.fit(X, y, epochs = 10000, shuffle = False)


import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

df = load_iris()
df.info()

df.keys()
x_data = np.array(df.data, dtype = np.float32)
y_data = np.array(df.target, dtype = np.float32)
x_data
y_data