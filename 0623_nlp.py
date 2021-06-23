# nltk 토큰화
import nltk
nltk.download('punkt')

sentence = "don't worry, be happy, please wake up everybody comes on"

from nltk.tokenize import word_tokenize
print(word_tokenize(sentence))
# ['do', "n't", 'worry', ',', 'be', 'happy', ',', 'please', 'wake', 'up', 'everybody', 'comes', 'on']

from nltk.tokenize import word_tokenize
word_tokenize(sentence)

from nltk.tokenize import wordpunct_tokenize
print(wordpunct_tokenize(sentence))


from konlpy.tag import *
kkma = Kkma()
komoran = Komoran()
okt = Okt()

sentence = '동해물과 백두산이 마르고 닳도록'
okt.nouns(sentence)
# ['해물', '백두산', '마르고']

okt.morphs(sentence)
# ['동', '해물', '과', '백두산', '이', '마르고', '닳도록']

okt.pos(sentence)
#  [('don', 'Alpha'),
#  ("'", 'Punctuation'),
#  ('t', 'Alpha'),
#  ('worry', 'Alpha'),
#  (',', 'Punctuation'),
#  ('be', 'Alpha'),
#  ('happy', 'Alpha'),
#  (',', 'Punctuation'),
#  ('please', 'Alpha'),
#  ('wake', 'Alpha'),
#  ('up', 'Alpha'),
#  ('everybody', 'Alpha'),
#  ('comes', 'Alpha'),
#  ('on', 'Alpha')]

kkma.nouns(sentence)

kkma.morphs(sentence)

kkma.pos(sentence)


# 문장 토큰화
text1 = 'Do you hear the people sing, Singing the song of angry men It is the music of the people who will not be slaves again'

from nltk.tokenize import sent_tokenize

sent_tokenize(text1)

text2 = '신종 코로나바이러스 감염증(코로나19) 여파로 각국 중앙은행이 초저금리 정책을 펼친데다 경기 부양으로 미국 등이 돈을 풀면서 풍부한 유동 자금이 부동산과 주식 등으로 유입, 전체적으로 오름세를 보인 것으로 풀이된다.'


import kss
kss.split_sentences(text2)
# 텍스트 정규화
# stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()

words = word_tokenize(text1)
[ps.stem(word) for word in words]

# lemmatization

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
wl = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have']
[wl.lemmatize(word) for word in words]

print(okt.morphs(text2))

print(okt.morphs(text2, stem=True))

print(okt.pos(text2))


# stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords  
print(stopwords.words('english'))

example = "drwill is very important person. It's everything"
stop_words = set(stopwords.words('english'))

word_tokenize(example)

word_tokens = word_tokenize(example)
result = []
for w in word_tokens:
    if w not in stop_words:
        result.append(w)

print('원문 :', word_tokens)
print('불용어 제거문:', result)

text2

stop_words = '미국 등이'
stop_words = stop_words.split(' ')
word_tokens = word_tokenize(text2)

result = [w for w in word_tokens if w not in stop_words]

print('원문 :',word_tokens)
print('불용어 제거 후 :', result)

# 정수 인코딩
text3 = ("I've been havin' dreams\
Jumpin' on a trampoline\
Flippin' in the air\
I never land, just float there\
As I'm looking up\
Suddenly the sky erupts\
Flames alight the trees\
Spread to fallin' leaves\
Now they're right upon me\
Wait if I'm on fire\
How am I so deep in love?\
When I dream of dying\
I never feel so loved\
I've been having dreams\
Splashin' in a summer stream\
Trip and I fall in\
I wanted it to happen\
My body turns to ice\
Crushin' weight of paradise\
Solid block of gold\
Lying in the cold\
I feel right at home\
Wait if I'm on fire\
How am I so deep in love?\
When I dream of dying\
I never feel so loved\
Wait if I'm on fire\
How am I so deep in love?\
When I dream of dying\
I never feel so loved\
I never feel so loved\
La, la, la, la, la\
La, la, la, la, la, la, la\
La, la, la, la, la\
La, la, la, la, la, la\
Wait if I'm on fire\
How am I so deep in love?\
When I dream of dying\
I never feel so loved\
Wait if I'm on fire\
How am I so deep in love?\
When I dream of dying\
I never feel so loved")

text = text3

text3 = sent_tokenize(text3)
print(text3)

# word tokenization
list(text3)
sent = []
stop_words = set(stopwords.words('english'))
for i in text3:
    sent = word_tokenize(i)
    result = []
    for w in sent:
        w = w.lower()
        if w not in stop_words:
            if len(w) > 2:
                result.append(w)
    sent.append(result)

print(sent)

for w in text3:
    w = w.lower()
    if w not in stop_words:
        if len(w) > 2:
            result.append(w)
sent.append(result)
text3

token = word_tokenize(text3)
token


text

text = sent_tokenize(text)
sentences = []

for i in text:
    sentence = word_tokenize(i)
    result = []
    for word in sentence:
        word = word.lower()
        if word not in stop_words:
            if len(word) >2:
                result.append(word)
    sentences.append(result)
print(sentences)
sentences

from collections import Counter
words = sum(sentences, [])
print(words)


text
sentesces = []
for i in text:
    sentence = word_tokenize(i)
    result = []
    for word in sentence:
        word = word.lower()
        if word not in stop_words:
            if len(word) >2:
                result.append(word)
    sentences.append(result)
print(sentences)

from collections import Counter
words = sum(sentences, [])
print(words)

Counter(words)

text

text = text3

sentence = 

words = sum(sentences,[])

vocab = Counter(words)

sorted(vocab.items(), key = lambda x: x[1])
sorted(vocab.items(), key = lambda x: x[1], reverse = True)

vocab_sorted = sorted(vocab.items(), key = lambda x: x[1], reverse = True)
vocab_sorted

word2idx = {}

i = 0

for (word, frequency) in vocab_sorted:
    if frequency > 1:
        i = i + 1
        word2idx[word] = i
print(word2idx)


vocab_size = 5
word_frequency = [w for w, c in word2idx.items() if c >= (vocab_size+1)]

for w in word_frequency:
    del word2idx[w]

print(word2idx)


word2idx