import nltk
nltk.download('gutenberg')
nltk.download('punkt')

from nltk.corpus import gutenberg
print(gutenberg.fileids())

emma = nltk.corpus.gutenberg.words('austen-emma.txt')
len(emma)

emma = nltk.Text(nltk.corpus.gutenberg.words('austen-emma.txt'))
emma.concordance("surprize")


emma = gutenberg.words('austen-emma.txt')


for fileid in gutenberg.fileids():
     num_chars = len(gutenberg.raw(fileid))
     num_words = len(gutenberg.words(fileid))
     num_sents = len(gutenberg.sents(fileid))
     num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)


for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab),fileid)

hamlet_sents = gutenberg.sents('shakespeare-hamlet.txt')
hamlet_sents
hamlet_sents[:500]

from nltk.tokenize import sent_tokenize
print(sent_tokenize(hamlet[:100]))

from nltk.tokenize import word_tokenize
print(word_tokenize(hamlet[:100]))

import nltk
nltk.download('averaged_perceptron_tagger')

from nltk.tag import pos_tag
sentence = 'You come most carefully vpon your houre'
tagged_list = pos_tag(word_tokenize(sentence))
print(tagged_list)
sentence2 = 'you must come back home'
tagged_list2 = pos_tag(word_tokenize(sentence2))
print(tagged_list2)

from nltk.tag import untag

untag(tagged_list)
untag(tagged_list2)

