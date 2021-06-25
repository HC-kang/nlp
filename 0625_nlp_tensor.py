# 자동 미분과 선형 회귀 실습
# 1. 자동 미분
import tensorflow as tf
import numpy as np


w = tf.Variable(2.)

def f(w):
    y = w**2
    z = 2*y + 5
    return z

with tf.GradientTape() as tape:
    z = f(w)

gradients = tape.gradient(z, [w])
print(gradients)


# 2. 자동 미분을 이용한 선형회귀 구현
W = tf.Variable(4.)
b = tf.Variable(1.)

@tf.function
def hypothesis(x):
    return W*x + b

x_test = [3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())


@tf.function
def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

X=[1,2,3,4,5,6,7,8,9] # 공부하는 시간
y=[11,22,33,44,53,66,77,87,95] # 각 공부하는 시간에 맵핑되는 성적

optimizer = tf.optimizers.SGD(0.01)

for i in range(301):
    with tf.GradientTape() as tape:
        
        y_pred = hypothesis(X)

        cost = mse_loss(y_pred, y)

    gradients = tape.gradient(cost, [W, b])

    optimizer.apply_gradients(zip(gradients, [W,b]))

    if i%10 ==0:
        print('epoch : {:3} | W의 값 : {:5.4f} | b의 값 : {:5.4} | cost : {:5.6f}'.format(i, W.numpy(), b.numpy(), cost))

x_test = [3.5, 5, 5.5, 6]
print(hypothesis(x_test).numpy())

# 케라스로 구현하는 선형회귀 - 안됨
model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_dim = 1))

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

X = [1,2,3,4,5,6,7,8,9]
y = [11,22,33,44,53,66,77,87,95]

model = Sequential()
model.add(Dense(1, input_dim = 1, activation = 'linear'))
sgd = optimizers.SGD(lr=0.01)

model.compile(optimizer = sgd, loss = 'mse', metrics = 'mse')

model.fit(X, y, batch_size = 1, epochs = 300, shuffle = False)

import matplotlib.pyplot as plt
plt.plot(X, model.predict(X), 'b', X, y, 'k.')

print(model.predict([9.5]))