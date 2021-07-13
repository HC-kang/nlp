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

    