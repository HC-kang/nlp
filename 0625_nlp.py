from nltk import tokenize
from nltk.text import TokenSearcher
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
import string 
import nltk
from tensorflow.python.keras.utils.np_utils import to_categorical

nltk.download('stopwords')

mail = pd.DataFrame(columns = ['num', 'label', 'text', 'label_num'])
mail['num'] = [1,2,3,4,5,6]
mail['label'] = ['spam', 'spam', 'ham', 'ham', 'ham', 'spam']
mail['text'] = ['your free lottery', 'free lottery free you', 'your free apple', 'free to contact me', 'I won award', 'my lottery ticket']
mail['label_num'] = [1,1,0,0,0,1]
mail


def process_text(text):
  # text에서 구두점(punctuation) 제거하기
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)
  # text에서 불용어 제거
  cleaned_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

  return cleaned_words


mail['text'].apply(process_text)


from sklearn.feature_extraction.text import CountVectorizer
msg_bow = CountVectorizer(analyzer= process_text).fit_transform(mail['text'])


# data split(train:8, test:2)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(msg_bow, mail['label_num'], test_size = 0.2, random_state=42)


# 다항식 나이브 베이즈 모델 훈련
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

classifier.predict(X_test)
y_test.values

classifier.predict(X_train)
y_train.values

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train_pred = classifier.predict(X_train) # 모델에 관한 예측값 출력
test_pred = classifier.predict(X_test) 

print(classification_report(y_train, train_pred))
print(classification_report(y_test, test_pred))

confusion_matrix(y_train, train_pred)
confusion_matrix(y_test, test_pred)

accuracy_score(y_train, train_pred)
accuracy_score(y_test, test_pred)

from konlpy.tag import *
okt = Okt()


token = okt.morphs('나는 자연어를 공부한다')
token

word2idx={}

for voca in token:
    if voca not in word2idx.keys():
        word2idx[voca] = len(word2idx)

print(word2idx)

def one_hot(word, word2idx):
    one_hot_vector = [0]*(len(word2idx))
    index = word2idx[word]
    one_hot_vector[index] = 1
    return one_hot_vector

one_hot('공부', word2idx)

from tensorflow.keras.preprocessing.text import Tokenizer

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print(tokenizer.word_index)

sub_text = '점심 먹으러 갈래 메뉴는 햄버거 최고야'
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

one_hot = to_categorical(encoded)
print(one_hot)
