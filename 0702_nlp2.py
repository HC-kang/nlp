import pandas as pd
import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/franciscadias/data/master/abcnews-date-text.csv", filename="abcnews-date-text.csv")
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)

text = data[['headline_text']]
text

import nltk
nltk.download('punkt')
text = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis = 1)

print(text.head())

from nltk.corpus import stopwords
stop = stopwords.words('english')

text = pd.DataFrame(text, columns = ['headline_text'])
text

text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop)])

from nltk.stem import WordNetLemmatizer
text['headline_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos= 'v') for word in x])

print(text.head())

tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word)>3])
print(tokenized_doc[:5])


# TF-IDF

detokenized_doc = [ ]
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

text['headline_text'] = detokenized_doc

text['headline_text'][:5]


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(stop_words = 'english', max_features=1000)

X = vect.fit_transform(text['headline_text'])
X.shape

# topic modeling
from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=10, learning_method = 'online', random_state = 777, max_iter=1)
lda_top = lda_model.fit_transform(X)

print(lda_model.components_)
print(lda_model.components_.shape)

terms = vect.get_feature_names()

def get_topics(components, feature_names, n = 5):
    for i, topic in enumerate(components):
        print('Topic %d:'% (i+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n-1:-1]])

get_topics(lda_model.components_, terms)


lda_model.components_