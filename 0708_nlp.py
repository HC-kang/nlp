
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

print(vector.fit_transform(corpus).toarray())

print(vector.vocabulary_)

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
          'you know I want your love',
          'I like you',
          'what should I do'
]

tfidfv = TfidfVectorizer().fit(corpus)


print(tfidfv.transform(corpus).toarray())


print(tfidfv.vocabulary_)


# abc뉴스 데이터로 TF-IDF
#####
import pandas as pd
import numpy as np
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

urllib.request.urlretrieve('https://raw.githubusercontent.com/franciscadias/data/master/abcnews-date-text.csv', filename='/content/abcnews-data-text.csv')

data = pd.read_csv('/content/abcnews-data-text.csv', error_bad_lines = False)
data

data.head()

text = data[['headline_text']]

text.nunique() # 고유값 수 출력

text.drop_duplicates(inplace=True)
text = text.reset_index(drop=True)
print(len(text))


# 데이터 정제 및 정규화
#####
# NLTK tokenizer를 통해 토큰화
text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis = 1)

stop_words = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop_words)])

text.head()

text['headline_text'] = text['headline_text'].apply(lambda x:[WordNetLemmatizer().lemmatize(word, pos='v') for word in x])

text = text['headline_text'].apply(lambda x:[word for word in x if len(word)>2])

print(text[:5])

detokenized_doc = []
for i in range(len(text)):
    t = ' '.join(text[i])
    detokenized_doc.append(t)

train_data = detokenized_doc

train_data[:5]

c_vectorizer = CountVectorizer(stop_words='english', max_features = 5000)
document_term_matrix = c_vectorizer.fit_transform(train_data)

print('행렬의 크기:', document_term_matrix.shape)

tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_features=5000)
tf_idf_matrix = tfidf_vectorizer.fit_transform(train_data)


print('행렬의 크기:', tf_idf_matrix.shape)


# !pip install beautifulsoup4
# !pip install newspaper3k
# !pip install konlpy


#!git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git


import konlpy
from konlpy.tag import Mecab
mecab = Mecab()

from bs4 import BeautifulSoup


html = '''
<html>
    <head>
    </head>
    <body>
        <h1>장바구니
            <p id='clothes' class='name' title='라운드티'>라운드티
                <span class = 'number'> 25 </span>
                <span class = 'price'> 29000 </span>
                <span class = 'menu'> 의류 </span>
                <a href = 'http://www.naver.com'> 바로가기 </a>
            </p>
            <p id = 'watch' class='name' title='시계'> 시계
                <span class='number'> 28 </span>
                <span class='price'> 32000 </span>
                <span class='menu'> 악세서리 </span>
                <a href = 'http://www.facebook.com' 바로가기 </a>
            </p>
        </h1>
    </body>
</html>
'''


soup = BeautifulSoup(html, 'html.parser')

print(soup.select('body'))

print(soup.select('h1 .name .menu'))

print(soup.select('html > h1'))


# Newspaper3k 패키지
#####
from newspaper import Article

url = 'https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=101&oid=030&aid=0002881076'

article = Article(url, language = 'ko')
article.download()
article.parse()

print('기사 제목:')
print(article.title)

print('기사 내용:')
print(article.text)


# BeautifulSoup와 newspaper3k를 통해 크롤러 만들기
#####
import requests
import pandas as pd
from bs4 import BeautifulSoup


def make_urllist(page_num, code, date):
    urllist = []
    for i in range(1, page_num + 1):
        url = 'https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=' + str(code) + '&date=' + str(date) + '&page' +str(i)
        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.90 Safari/537.36'}
        news = requests.get(url, headers=headers)

        soup = BeautifulSoup(news.content, 'html.parser')

        # case1
        news_list = soup.select('.newsflash_body .type06_headline li dl')

        # case2
        news_list.extend(soup.select('.newsflash_body .type06 li dl'))

        for line in news_list:
            urllist.append(line.a.get('href'))
            # 각 뉴스로부터 a 태그인 <a href='주소'> 에서 '주소'만 가져오기
    return urllist

url_list = make_urllist(3, 101, 20200506)
print('뉴스 기사의 갯수:', len(url_list))

url_list[:5]

idx2word = {'101': '경제', '102':'사회', '103':'생활/문화', '105':'IT/과학'}


from newspaper import Article

def make_data(urllist, code):
    text_list = []
    for url in urllist:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        text_list.append(article.text)

    df = pd.DataFrame({'news': text_list})

    df['code'] = idx2word[str(code)]
    return df

data = make_data(url_list, 101)
data[:10]


# 데이터 수집 및 전처리
#####
code_list = [102, 103, 105]
code_list

def make_total_data(page_num, code_list, date):
    df = None

    for code in code_list:
        url_list = make_urllist(page_num, code, date)
        df_temp = make_data(url_list, code)
        print(str(code) + '번 코드에 대한 데이터를 만들었습니다.')

        if df is not None:
            df = pd.concat([df, df_temp])
        else:
            df = df_temp

    return df


df = make_total_data(1, code_list, 20200506)


print('뉴스 기사의 갯수:', len(df))


df.sample(10)


df = make_total_data(100, code_list, 20200506)


from konlpy.tag import Mecab
mecab = Mecab()

from nltk.corpus import abc

import nltk
nltk.download('abc')
nltk.download('punkt')

corpus = abc.sents()

print(corpus[:3])

print('코퍼스의 크기:', len(corpus))

from gensim.models import Word2Vec

model = Word2Vec(sentences = corpus, size = 100, window = 5, min_count = 5, workers = 4, sg = 0)

model_result = model.wv.most_similar('man')

print(model_result)

from gensim.models import KeyedVectors

model.wv.save_word2vec_format('./w2v')
loaded_model = KeyedVectors.load_word2vec_format('./w2v')
print('모델 load 완료')


model_result = loaded_model.wv.most_similar('man')


print(model_result)

loaded_model.wv.most_similar('memory')

import os
os.chdir('/content/drive/MyDrive/AI_data')
csv_path = './news_data.csv'

df = pd.read_csv(csv_path)

df['news'] = df['news'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣]", "")
df['news']

print(df.isna().sum())

len(df)

df.iloc[[2006, 2009],:]


df.drop_duplicates(subset=['news'], inplace=True)
print('뉴스 기사의 갯수 :', len(df))


# 데이터 탐색
#####
df['code'].value_counts().plot(kind='bar')

print(df.groupby('code').size().reset_index(name='count'))

tokenizer = Mecab()

kor_text = '밤에 귀가하던 여성에게 범죄를 시도한 대 남성이 구속됐다서울 제주경찰서는 \
            상해 혐의로 씨를 구속해 수사하고 있다고 일 밝혔다씨는 지난달 일 피해 여성을 \
            인근 지하철 역에서부터 따라가 폭행을 시도하려다가 도망간 혐의를 받는다피해 \
            여성이 저항하자 놀란 씨는 도망갔으며 신고를 받고 주변을 수색하던 경찰에 \
            체포됐다피해 여성은 이 과정에서 경미한 부상을 입은 것으로 전해졌다'


print(tokenizer.morphs(kor_text))

stopwords = ['에','는','은','을','했','에게','있','이','의','하','한','다','과','때문','할','수','무단','따른','및','금지','전재','경향신문','기자','는데','가','등','들','파이낸셜','저작','등','뉴스']

def preprocessing(data):
    text_data = []

    for sentence in data:
        temp_data = []

        temp_data = tokenizer.morphs(sentence)

        temp_data = [word for word in temp_data if not word in stopwords]
        text_data.append(temp_data)

    text_data = list(map(' '.join, text_data))

    return text_data


text_data = preprocessing(df['news'])

print(text_data[0])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

print(df.head())
print(df.shape)

x_train, x_test, y_train, y_test = train_test_split(text_data, df['code'], random_state = 0)

print('훈련용 뉴스 기사의 수:', len(x_train))
print('테스트용 뉴스 기사의 수:', len(x_test))
print('훈련용 레이블 수:', len(y_train))
print('테스트용 레이블 수:', len(y_test))


count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

clf = MultinomialNB().fit(x_train_tfidf, y_train)

def tfidf_vectorizer(data):
    data_counts = count_vect.transform(data)
    data_tfidf = tfidf_transformer.transform(data_counts)
    return data_tfidf

new_sent = preprocessing(["민주당 일각에서 법사위의 체계·자구 심사 기능을 없애야 한다는 \
                           주장이 나오는 데 대해 “체계·자구 심사가 법안 지연의 수단으로 \
                          쓰이는 것은 바람직하지 않다”면서도 “국회를 통과하는 법안 중 위헌\
                          법률이 1년에 10건 넘게 나온다. 그런데 체계·자구 심사까지 없애면 매우 위험하다”고 반박했다."])
print(clf.predict(tfidf_vectorizer(new_sent)))

new_sent = preprocessing(["인도 로맨틱 코미디 영화 <까립까립 싱글>(2017)을 봤을 때 나는 두 눈을 의심했다. \
                          저 사람이 남자 주인공이라고? 노안에 가까운 이목구비와 기름때로 뭉친 파마머리와, \
                          대충 툭툭 던지는 말투 등 전혀 로맨틱하지 않은 외모였다. 반감이 일면서 \
                          ‘난 외모지상주의자가 아니다’라고 자부했던 나에 대해 회의가 들었다.\
                           티브이를 꺼버릴까? 다른 걸 볼까? 그런데, 이상하다. 왜 이렇게 매력 있지? 개구리와\
                            같이 툭 불거진 눈망울 안에는 어떤 인도 배우에게서도 느끼지 못한 \
                            부드러움과 선량함, 무엇보다 슬픔이 있었다. 2시간 뒤 영화가 끝나고 나는 완전히 이 배우에게 빠졌다"])
print(clf.predict(tfidf_vectorizer(new_sent)))

y_pred = clf.predict(tfidf_vectorizer(x_test))
print(metrics.classification_report(y_test, y_pred))