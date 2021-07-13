## LSA 실습

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state = 1, remove=('headers', 'footers', 'quotes'))

documents = dataset.data

d = {'document':documents}
news_df = pd.DataFrame(data = d)
news_df['clean_doc'] = news_df['document'].str.replace('[^a-zA-Z]', ' ')
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join(w for w in x.split() if len(w)>3))
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
news_df

import nltk
nltk.download('all')

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
news_df['clean_doc'] = detokenized_doc

news_df

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english', max_features = 1000, max_df = 0.5, smooth_idf = True)
X = vectorizer.fit_transform(news_df['clean_doc'])

from sklearn.decomposition import TruncatedSVD

svd_model = TruncatedSVD(n_components=20, algorithm = 'randomized', n_iter = 100, random_state = 122)
svd_model.fit(X)

terms = vectorizer.get_feature_names()
def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print('Topic %d:' %(idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n-1:-1]])

get_topics(svd_model.components_, terms)



## LDA 실습

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state = 1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

news_df = pd.DataFrame({'document' :documents})
news_df['clean_doc'] = news_df['document'].str.replace('[^a-zA-Z]', ' ')
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join(w for w in x.split() if len(w) >3))
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
news_df

import nltk
nltk.download('all')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
tokenized_doc = news_df['clean_doc'].apply(lambda x:x.split())
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

from gensim import corpora
dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]

import gensim
num_topics = 20
k = 20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
topics = ldamodel.print_topics(num_words=4)

for topic in topics:
  print(topic)

  print('Perplexity: ', ldamodel.log_perplexity(corpus))

from gensim.models.coherencemodel import CoherenceModel
coherence_model_lda = CoherenceModel(model=ldamodel, texts = tokenized_doc, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score:', coherence_lda)

print(ldamodel.show_topics(formatted=True))


# 이미지에서 글자 추출하기

pip install -upgrade pip

from pytesseract import pytesseract
from PIL import Image

def load_image(path, mode = ''):
    return Image.open(path)

def run_pytesseract(image, path_engine, config='-l eng'):
    pytesseract.tesseract_cmd = path_engine
    result = pytesseract.image_to_string(image, config=config)
    return result

import pytesseract
import cv2
import os
from PIL import Image
from google.colab.patches import cv2_imshow

!sudo apt install tesseract-ocr

image = cv2.imread('/content/drive/MyDrive/AI_data/test.jpg')

image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

filename = '{}.png'.format(os.getpid())
cv2.imwrite(filename, gray)

filename = '{}.png'.format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename), lang=None)
os.remove(filename)

print(text)

cv2_imshow(image)

## PDF 에서 글 추출하기

!pip install pdfminer

from io import StringIO
import pdfminer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

def convert_pdf2txt(path):
    manager = PDFResourceManager()
    file_object = StringIO()
    converter = TextConverter(manager, file_object, laparams=LAParams())

    with open(path, 'rb') as f:
        interpreter = PDFPageInterpreter(manager, converter)
        for page in PDFPage.get_pages(f, maxpages=0, caching=True, check_extractable=True):
            interpreter.process_page(page)
    
    converter.close()
    text = file_object.getvalue()
    file_object.close()
    return text

convert_pdf2txt('/content/drive/MyDrive/AI_data/sample_papers.pdf')

## PDF에서 글 추출 2번째

!pip install pdfplumber

import pdfplumber

text = ''
with pdfplumber.open('/content/drive/MyDrive/AI_data/sample_papers.pdf') as pdf:
    for page in pdf.pages:
        text += page.extract_text()
    
text

## 한국어 word2vec 만들기(네이버 영화 리뷰)

pip install konlpy

import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings.txt", filename="ratings.txt")

train_data = pd.read_table('/content/ratings.txt')

train_data[:5]

print(len(train_data))

print(train_data.isna().values.any())

train_data = train_data.dropna(how='any')

print(train_data.isna().values.any())

print(len(train_data))

train_data['document'] = train_data['document'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣]', ' ')

train_data

stopwords = ['의' ,'가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

okt = Okt()
tokenized_data = []
for sentence in train_data['document']:
    temp_x = okt.morphs(sentence, stem=True)
    temp_x = [word for word in temp_x if not word in stopwords]
    tokenized_data.append(temp_x)



import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter


urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")

train_data = pd.read_table('ratings_train.txt')
test_data = pd.read_table('ratings_test.txt')
train_data.head()

from konlpy.tag import Mecab

!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git
!cd Mecab-ko-for-Google-Colab/
!bash install_mecab-ko_on_colab190912.sh

tokenizer = Mecab()

def tokenize_and_remove_stopwords(data, stopwords, tokenizer):
    result = []

    for sentence in data:
        curr_data = []
        curr_data = tokenizer.morphs(sentence) #토큰화
        curr_data = [word for word in curr_data if not word in stopwords]

        result.append(curr_data)
    return result

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

def load_data(train_data, test_data, num_words=10000):
    train_data.drop_duplicates(subset=['document'], inplace=True)
    test_data.drop_duplicates(subset=['document'], inplace=True)

    train_data = train_data.dropna(how='any')
    test_data = test_data.dropna(how='any')

    x_train = tokenize_and_remove_stopwords(train_data['document'], stopwords, tokenizer)
    x_test = tokenize_and_remove_stopwords(test_data['document'], stopwords, tokenizer)

    words = np.concatenate(x_train).tolist()
    counter = Counter(words)
    counter = counter.most_common(10000-4)
    vocab = ['<PAD>','<BOS>','<UNK>','<UNUSED>'] + [key for key, _ in counter]
    word_to_index = {word:index for index, word in enumerate(vocab)}

    def wordlist_to_lndexlist(wordlist):
        return [word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in wordlist]

    x_train = list(map(wordlist_to_lndexlist, x_train))
    x_test = list(map(wordlist_to_lndexlist, x_test))

    return x_train, np.array(list(train_data['label'])), x_test, np.array(list(test_data['label'])),word_to_index

x_train, y_train, x_test, y_test, word_to_index = load_data(train_data, test_data)
print(x_train[10])


words = np.concatenate(x_train).tolist()
counter = Counter(words)
counter = counter.most_common(10000-4)

print(counter)

index_to_word = {index:word for word, index in word_to_index.items()}

def get_encoded_sentence(sentence, word_to_index):
    return [word_to_index['<BOS>']]+[word_to_index[word] if word in word_to_index else word_to_index['<UNK>'] for word in sentence.split()]

def get_encoded_sentences(sentences, word_to_index):
    return [get_encoded_sentence(sentence, word_to_index) for sentence in sentences]

def get_decoded_sentence(encoded_sentence, index_to_word):
    return ' '.join(index_to_word[index] if index in index_to_word else '<UNK>' for index in encoded_sentence[1:])

def get_decoded_sentences(encoded_sentences, index_to_word):
    return [get_decoded_sentence(encoded_sentence, index_to_word) for encoded_sentence in encoded_sentences]

get_decoded_sentence(x_train[10], index_to_word)

## 모델 구성을 위한 데이터 분석 및 가공


total_data_text = list(x_train) + list(x_test)
num_tokens = [len(tokens) for tokens in total_data_text]
num_tokens = np.array(num_tokens)
print('문장길이 평균:', np.mean(num_tokens))
print('문장길이 최대:', np.max(num_tokens))
print('문장길이 표준편차:', np.std(num_tokens))

## 최대길이 = (평균 + 2*표준편차)
max_tokens = np.mean(num_tokens) + 2* np.std(num_tokens)
maxlen = int(max_tokens)
print('pad_sequences maxlen:', maxlen)
print('전체 문장의 {}%가 maxlen 설정값 이내에 포함됩니다.'.format(100*np.sum(num_tokens < max_tokens)/len(num_tokens)))

x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=word_to_index['<PAD>'], padding='pre', maxlen = maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, value=word_to_index['<PAD>'], padding='pre', maxlen = maxlen)

print(x_train.shape)
print(x_test.shape)

## 모델 구성 및 validation 구성


vocab_size = 10000
word_vector_dim = 200 # 워드 벡터의 차원 수
'''
RNN 버전
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
'''
##1D-CNN

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, word_vector_dim, input_shape=(None,)))
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.MaxPool1D(5))
model.add(keras.layers.Conv1D(16, 7, activation='relu'))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(8, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))


model.summary()

## 모델 훈련

x_val = x_train[:50000]
y_val = y_train[:50000]

partial_x_train = x_train[50000:]
partial_y_train = y_train[50000:]

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = 'accuracy')

epochs = 15

history = model.fit(partial_x_train, partial_y_train, epochs = epochs, batch_size = 512, validation_data=(x_val, y_val), verbose = 1)

result = model.evaluate(x_test, y_test, verbose=2)
print(result)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label = 'Trainig loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

## 학습된 embedding 레이어 분석

import os
word2vec_file_path = 'word2vec.txt'
f = open(word2vec_file_path, 'w')
f.write('{} {} \n'.format(vocab_size-4, word_vector_dim))

vectors = model.get_weights()[0]
for i in range(4, vocab_size):
  f.write('{} {}\n'.format(index_to_word[i], ' '.join(map(str, list(vectors[i, :])))))
f.close()

from gensim.models.keyedvectors import Word2VecKeyedVectors

word_vector = Word2VecKeyedVectors.load_word2vec_format(word2vec_file_path, binary=False)
vector = word_vector['짜증']
vector

word_vector.similar_by_word("짜증")

word_vector.similar_by_word("재미")

### 한국어 word2vec임베딩을 활용해서 성능 개선

import gensim

word2vec_path = '/content/drive/MyDrive/자연어강의안/영우_2021_3반_자연어/dataset/ko.bin'
word2vec = gensim.models.Word2Vec.load(word2vec_path)
vector = word2vec['감동']
vector

word2vec.similar_by_word('재미')

word2vec.similar_by_word('로맨틱')

## 네이버 쇼핑 리뷰 검색

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

### 데이터 로드

urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")

total_data = pd.read_table('ratings_total.txt', names=['ratings', 'reviews'])
print('전체 리뷰 갯수 :', len(total_data))

total_data[:5]

### 훈련데이터와 테스트데이터를 분리

total_data['label'] = np.select([total_data.ratings >3], [1], default=0)
total_data[:5]

total_data['ratings'].nunique(), total_data['reviews'].nunique(), total_data['label'].nunique()

total_data.drop_duplicates(subset=['reviews'], inplace=True)
print('샘플의 수:', len(total_data))

print(total_data.isna().values.any())

train_data, test_data = train_test_split(total_data, test_size = 0.25, random_state = 42)
print('훈련용 리뷰의 개수:', len(train_data))
print('테스트용 리뷰의 개수:', len(test_data))

## 레이블의 분포 확인

train_data['label'].value_counts().plot(kind = 'bar')

print(train_data.groupby('label').size().reset_index(name='count'))

## 데이터 정제하기

train_data['reviews'] = train_data['reviews'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣]', '')
train_data['reviews'].replace('', np.nan, inplace = True)
print(train_data.isna().sum())

test_data.drop_duplicates(subset=['reviews'], inplace=True)
test_data['reviews'] = test_data['reviews'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]","")
test_data['reviews'].replace('',np.nan, inplace=True)
test_data= test_data.dropna(how='any')
print('전처리 후 테스트용 샘플의 갯수: ', len(test_data))

## 토큰화


mecab = Mecab()
print(mecab.morphs('이런 상품도 상품이라고 허허허'))

## 불용어 제거

stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

train_data['tokenized'] = train_data['reviews'].apply(mecab.morphs)
train_data['tokenized'] = train_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

test_data['tokenized'] = test_data['reviews'].apply(mecab.morphs)
test_data['tokenized'] = test_data['tokenized'].apply(lambda x: [item for item in x if item not in stopwords])

## 단어와 길이 분포 확인하기

negative_words = np.hstack(train_data[train_data.label==0]['tokenized'].values)
positive_words = np.hstack(train_data[train_data.label==1]['tokenized'].values)

negative_word_count = Counter(negative_words)
print(negative_word_count.most_common(20))

positive_word_count = Counter(positive_words)
print(positive_word_count.most_common(20))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
text_len = train_data[train_data['label']==1]['tokenized'].map(lambda x: len(x))
ax1.hist(text_len, color='red')
ax1.set_title('Positive Reviews')
ax1.set_xlabel('length of samples')
ax1.set_ylabel('number of samples')
print('긍정 리뷰의 평균 길이 :', np.mean(text_len))

text_len = train_data[train_data['label']==0]['tokenized'].map(lambda x: len(x))
ax2.hist(text_len, color='blue')
ax2.set_title('Negative Reviews')
ax2.set_xlabel('length of samples')
ax2.set_ylabel('number of samples')
print('부정 리뷰의 평균 길이 :', np.mean(text_len))
plt.show()

train_data.head()

x_train = train_data['tokenized'].values
y_train = train_data['label'].values
x_test = test_data['tokenized'].values
y_test = test_data['label'].values

## 정수 인코딩

t = Tokenizer()
t.fit_on_texts(x_train)

threshold = 2
total_cnt = len(t.word_index) 
rare_cnt = 0
total_freq = 0
rare_freq = 0

for key, value in t.word_counts.items():
  total_freq = total_freq + value

  if (value < threshold):
    rare_cnt = rare_cnt + 1
    rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :', total_cnt)
print('등장 빈도가 %s번 이하인 희귀단어의 수 : %s'%(threshold-1, rare_cnt))
print('단어 집합에서 희귀단어의 비율 :', (rare_cnt/total_cnt)*100)
print('전체 등장 빈도에서 희귀단어 등장 빈도 비율 :',(rare_freq/total_freq)*100)

vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기:', vocab_size)

original_vocab_size = vocab_size + rare_cnt -2
print('원래 vocab_size:', original_vocab_size)

tokenizer = Tokenizer(vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)

print(x_train[:3])
print(x_test[:3])


## 패딩


print('리뷰의 최대 길이:', max(len(l) for l in x_train))
print('리뷰의 평균 길이 :', sum(map(len, x_train))/len(x_train))
plt.hist([len(s) for s in x_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
      cnt = 0
  for s in nested_list:
    if (len(s) <= max_len):
      cnt = cnt+1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt/len(nested_list))))

max_len = 80
below_threshold_len(max_len, x_train)

x_train = pad_sequences(x_train, maxlen = max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

print(x_train.shape)
print(x_test.shape)

## GRU 모델 학습

from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(GRU(128))
model.add(Dense(1, activation = 'sigmoid'))

es = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience = 4)
mc = ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', verbose = 1, save_best_only = True)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=30, callbacks=[es, mc], batch_size=60, validation_split=0.2)

loaded_model = load_mode('best_model.h5')
print('테스트 정확도: %.4f'%(loaded_mode.evaluate(x_test, y_test[1])))

## 리뷰 예측하기

def sentiment_predict(new_sentence):
    new_sentence = mecab.morphs(new_sentence)
    new_sentence = [word for word in new_sentence if not word in stopwords]
    encoded = tokenizer.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen = max_len)

    score = float(load_model.predict(pad_new))

    if (score> 0.5):
        print('{:.2f}% 확률로 긍정 리뷰입니다.'.format(score*100))
    else:
        print('{:.2f} % 확률로 부정 리뷰입니다. '.format((1-score)*100))