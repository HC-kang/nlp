#####
# NLP 1강
###

# Word Tokenization (영어)
import nltk
from nltk.util import pad_sequence
from yaml.tokens import Token
nltk.download('punkt')

sentence = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."

from nltk.tokenize import word_tokenize
print(word_tokenize(sentence))
#['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

from nltk.tokenize import WordPunctTokenizer
print(WordPunctTokenizer().tokenize(sentence))
#['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']

from nltk.tokenize import TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()
text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))
# ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']

''' Word Tokenization (한국어)'''


from konlpy.tag import *

hannanum = Hannanum()
kkma = Kkma()
komoran = Komoran()
okt = Okt()
mecab = Mecab()

print(okt.nouns('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(okt.morphs('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(okt.pos('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))

print(kkma.nouns('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(kkma.morphs('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(kkma.pos('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))

print(komoran.nouns('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(komoran.morphs('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(komoran.pos('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))

print(hannanum.nouns('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(hannanum.morphs('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(hannanum.pos('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))

print(mecab.nouns('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(mecab.morphs('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))
print(mecab.pos('열심히 코딩한 당신, 연휴에는 여행을 가봐요'))

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."

from nltk.tokenize import sent_tokenize
print(sent_tokenize(text))

text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))

import kss
text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어려워요. 농담아니에요...!.이제 해보면 알걸요?'
print(kss.split_sentences(text))

from nltk.stem import PorterStemmer
porterstemmer = PorterStemmer()

text = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
words = word_tokenize(text)
print(words)

print([porterstemmer.stem(w) for w in words])

words = ['formalize', 'allowance', 'electricical']
print([porterstemmer.stem(w) for w in words])

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
wordnetlemmatizer = WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print([wordnetlemmatizer.lemmatize(w) for w in words])

text = "북한은 하루새 3차례에 걸쳐 대미·대남 압박 메시지를 내놓았다."
print(okt.morphs(text))
print(okt.morphs(text, stem=True))
print(okt.pos(text))
print(okt.pos(text, stem=True))

text = "웃기는 소리하지마랔ㅋㅋㅋ"
print(okt.morphs(text))
print(okt.morphs(text, norm=True))
print(okt.pos(text))
print(okt.pos(text, norm=True))

nltk.download('stopwords')

from nltk.corpus import stopwords
print(stopwords.words('english'))

example = 'Family is not an important thing. It\'s evertything'
stop_words = set(stopwords.words('english'))

word_token = word_tokenize(example)

result = []
for w in word_token:
    if w not in stop_words:
        result.append(w)
    
print('원문 :', word_token)
print('불용어 제거 후 :', result)

# Integer Encoding

text = "A barber is a person. a barber is good person. a barber is huge person. \
        he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. \
        His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. \
        the barber went up a huge mountain."

text = sent_tokenize(text)
print(text)

sentences = []
stop_words = set(stopwords.words('english'))

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

vocab = Counter(words)
print(vocab)

print(vocab['barber'])

vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
print(vocab_sorted)

word2idx = {}
i=0
for (word, frequency) in vocab_sorted:
    if frequency > 1:
        i += 1
        word2idx[word] = i
print(word2idx)

vocab_size = 5
words_frequency = [w for w, c in word2idx.items() if c >= vocab_size +1]
for w in words_frequency:
    del word2idx[w]

print(word2idx)

word2idx['OOV'] = len(word2idx) + 1
print(word2idx)

encoded = []
for s in sentences:
    temp = []
    for w in s:
        try:
            temp.append(word2idx[w])
        except:
            temp.append(word2idx['OOV'])
    encoded.append(temp)
encoded

print(sentences)

from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

print(tokenizer.word_index)

print(tokenizer.word_counts)

print(tokenizer.texts_to_sequences(sentences))


vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size +1)
tokenizer.fit_on_texts(sentences)

print(tokenizer.texts_to_sequences(sentences))

vocab_size = 5
tokenizer = Tokenizer(num_words = vocab_size +2, oov_token = 'OOV')
tokenizer.fit_on_texts(sentences)
print(tokenizer.texts_to_sequences(sentences))

print('단어 OOV의 인덱스 : {}'.format(tokenizer.word_index['OOV']))

print(tokenizer.texts_to_sequences(sentences))

import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], 
             ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'],
             ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'],
             ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
             ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

encoded = tokenizer.texts_to_sequences(sentences)
print(encoded)

max_len = max(len(item) for item in encoded)
print(max_len)

for item in encoded:
    while len(item) < max_len:
        item.append(0)

padded_np = np.array(encoded)

padded_np


encoded = tokenizer.texts_to_sequences(sentences)
print(encoded)

padded = pad_sequences(encoded)

padded

padded = pad_sequences(encoded, padding = 'post')
padded

(padded ==padded_np).all()

padded = pad_sequences(encoded, padding = 'post', maxlen=5)
padded

last_value = len(tokenizer.word_index) + 1
print(last_value)

padded = pad_sequences(encoded, padding = 'post', value = last_value)
padded

token = okt.morphs('나는 자연어 처리를 배운다')
print(token)
word2idx={}
for voca in token:
    if voca not in word2idx.keys():
        word2idx[voca] = len(word2idx)
print(word2idx)

def one_hot_encoding(word, word2idx):
    one_hot_vector = [0] * len(word2idx)
    index = word2idx[word]
    one_hot_vector[index] = 1
    return one_hot_vector

one_hot_encoding('자연어', word2idx)

from tensorflow.keras.utils import to_categorical
text = '나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야'

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
print(tokenizer.word_index)

sub_text = '점심 먹으러 갈래 메뉴는 햄버거 최고야'
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

one_hot = to_categorical(encoded)
print(one_hot)

