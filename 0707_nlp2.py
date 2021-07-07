import re, collections

num_merges = 10 # BPE를 몇 번 수행할 것인지.

dictionary = {'l o w </w>': 5, 
              'l o w e r </w>': 2, 
              'n e w e s t </w>': 6, 
              'w i d e s t </w>': 3
              }

def get_stats(dictionary):
    # 유니그램의 pair들의 빈도수를 카운트
    pairs = collections.defaultdict(int)
    for word, freq in dictionary.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq
    print('현재 pair들의 빈도수 :', dict(pairs))
    return pairs

def merge_dictionary(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

bpe_codes = {}
bpe_codes_reverse = {}
for i in range(num_merges):
    print(">> Step {0}".format(i+1))
    pairs = get_stats(dictionary)
    best = max(pairs, key=pairs.get)
    dictionary = merge_dictionary(best, dictionary)

    bpe_codes[best] = i
    bpe_codes_reverse[best[0] + best[1]] = best

    print("new merge: {}".format(best))
    print("dictionary: {}".format(dictionary))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

orig = 'hi'
word = tuple(orig) + ('</w>',)
print(word)

def encode(orig):
    word = tuple(orig) + ('</w>',)
    print('__word split into characters:__ <tt>{}<tt>'.format(word))

    pairs = get_pairs(word)

    if not pairs:
        return orig
    
    iteration = 0
    while True:
        iteration += 1
        print('__Iteration {}:__'.format(iteration))
        
        print('Bigram in the word: {}'.format(pairs))
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        print('candidate for merging: {}'.format(bigram))
        if bigram not in bpe_codes:
            print('__Cnadidate not in BPE merges, algorithm stops.__')
            break
        first, second = bigram
        new_word = []
        i = 0
        while i<len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i=j
            except:
                new_word.extend(word[i:])
                break
            
            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2 
            else:
                new_word.append(word[i])
                i += 1 
        new_word = tuple(new_word)
        word = new_word
        print('word after merging: {}'.format(word))
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    
    # 특별 토큰인 </w>는 출력하지 않는다.
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>', ''),)
    return word

encode("loki")
encode("lowing")
encode("highing")

#####################

import tensorflow_datasets as tfds
import urllib.request
import pandas as pd

urllib.request.urlretrieve('https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv', filename='IMDb_Reviews.csv')

train_df = pd.read_csv('IMDb_Reviews.csv')

train_df

train_df['sentiment']

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(train_df['review'], target_vocab_size=2**13)

print(tokenizer.subwords[:100])

print(tokenizer.subwords[-100:])

print(train_df['review'][20])

print('토큰화된 샘플 질문:{}'.format(tokenizer.encode(train_df['review'][20])))

# 리뷰데이터가 아닌 샘플 문장으로 인코딩하고 디코딩 하기
sample_string1 = "It's mind-blowing to me that this film was even made."

# 인코딩 해서 저장
tokenized_string1 = tokenizer.encode(sample_string1)
print('정수 인코딩 후의 문장: {}'.format(tokenized_string1))

# 이것을 다시 디코딩해서 저장
original_string1 = tokenizer.decode(tokenized_string1)
print('기존 문장 복구: {}'.format(original_string1))

print('단어 집합의 크기(Vocab size):', tokenizer.vocab_size)

for ts in tokenized_string1:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))

sample_string = "It's mind-blowing to me that this film was evenxyz made.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

tokenized_string = tokenizer.encode(sample_string)
print('인코딩 전 문장:{}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('기존 문장 복구: {}'.format(original_string))

for ts in tokenized_string:
    print('{} ----> {}'.format(ts, tokenizer.decode([ts])))


##############################
# !pip install sentencepiece

import sentencepiece as spm
import pandas as pd
import urllib.request
import csv

train_df = pd.read_csv('IMDb_Reviews.csv')
train_df

train_df['review']

print('리뷰 개수:', len(train_df['review']))

with open('imdb_review.txt', 'w', encoding = 'utf-8') as f:
    f.write('\n'.join(train_df['review']))

spm.SentencePieceTrainer.Train('--input=imdb_review.txt --model_prefix=imdb --vocab_size=5000 --model_type=bpe --max_sentence_length=9999')

vocab_list = pd.read_csv('imdb.vocab', sep='\t', header=None, quoting=csv.QUOTE_NONE)

len(vocab_list)

sp = spm.SentencePieceProcessor()
vocab_file = 'imdb.model'
sp.load(vocab_file)

lines = [
         "I didn't at all think of it this way",
         "I have waited a long time for someone to film",
]
for line in lines:
    print(line)
    print(sp.encode_as_pieces(line)) # 문장을 입력하면 subword sequence로 변경
    print(sp.encode_as_ids(line)) # 문장을 입력하면 정수 시퀀스로 변경

sp.GetPieceSize()

sp.IdToPiece(430)

sp.PieceToId('▁character') # 정수로부터 매핑되는 서브워드로 변환

sp.DecodeIds([41, 141, 1364, 1120,4,666,285, 92, 1078, 33, 91])
# 정수 시퀀스로부터 문장으로 변환

# 서브워드 시퀀스로부터 문장으로 변환
sp.DecodePieces(['_I', '_have', '_wa', 'ited', '_a', '_long', '_time', '_for', '_someone', '_to', '_film'])

print(sp.encode('I have waited a long time for someone to film', out_type = str))
print(sp.encode('I have waited a long time for someone to film', out_type = int))
# encode -> 문장으로부터 인자값에 따라서 정수 시퀀스 또는 서브워드 시퀀스로 변환이 가능함.

