# 노이즈 유형1 구두점 제거
def pad_punctuation(sentence, punc):
    for p in punc:
        sentence = sentence.replace(p, ' '+p+' ')
    return sentence

sentence = 'Hi, my name is john.'

print(pad_punctuation(sentence, ['.', '?', '!', ',']))

# 노이즈 유형2 모든 단어를 소문자로 바꾸는 방법

sentence = 'First, open the first chapter.'
print(sentence.lower())

print(sentence.upper())

# 노이즈 유형3 특수문자 제거
import re
sentence = 'He is a ten-year-old boy'
sentence = re.sub('([^a-zA-Z.,?!])', ' ', sentence)

print(sentence)


# 배운 것들을 종합하기

corpus = '''
The sheds where the corn was stored, the stable where the horses were kept, and the yard where the cows were milked morning and evening were unfailing sources of interest to Martha and me. The milkers would let me keep my hands on the cows while they milked, and I often got well switched by the cow for my curiosity.

The making ready for Christmas was always a delight to me. Of course I did not know what it was all about, but I enjoyed the pleasant odours that filled the house and the tidbits that were given to Martha Washington and me to keep us quiet. We were sadly in the way, but that did not interfere with our pleasure in the least. They allowed us to grind the spices, pick over the raisins and lick the stirring spoons. I hung my stocking because the others did; I cannot remember, however, that the ceremony interested me especially, nor did my curiosity cause me to wake before daylight to look for my gifts.

Martha Washington had as great a love of mischief as I. Two little children were seated on the veranda steps one hot July afternoon. One was black as ebony, with little bunches of fuzzy hair tied with shoestrings sticking out all over her head like corkscrews. The other was white, with long golden curls. One child was six years old, the other two or three years older. The younger child was blind—that was I—and the other was Martha Washington. We were busy cutting out paper dolls; but we soon wearied of this amusement, and after cutting up our shoestrings and clipping all the leaves off the honeysuckle that were within reach, I turned my attention to Martha's corkscrews. She objected at first, but finally submitted. Thinking that turn and turn about is fair play, she seized the scissors and cut off one of my curls, and would have cut them all off but for my mother's timely interference.
'''

corpus2 = '''
My studies the first year were French, German, history, English composition and English literature. In the French course I read some of the works of Corneille, Moliere, Racine, Alfred de Musset and Sainte-Beuve, and in the German those of Goethe and Schiller. I reviewed rapidly the whole period of history from the fall of the Roman Empire to the eighteenth century, and in English literature studied critically Milton's poems and "Areopagitica."
'''

import re

def cleaning_text(text, punc, regex):
    text = text.lower()
    for p in punc:
        text = text.replace(p, ' '+p+' ')
    text = re.sub(regex, ' ', text)
    return text

print(cleaning_text(corpus, [',','.','!','?'], '([^a-zA-Z0-9.,?!\n])'))
print(cleaning_text(corpus2, [',','.','!','?'], '([^a-zA-Z0-9.,?!\n])'))




#########################

# !git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git

# !bash install_mecab-ko_on_colab190912.sh

# !pip install konlpy

from konlpy.tag import Mecab

mecab = Mecab()

print(mecab.morphs('자연어처리가너무재밌어서밥먹는것도가끔까먹어요'))
print(mecab.morphs('집에가고싶다'))