import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk import PorterStemmer
from gensim import corpora
from gensim import models
from nltk.stem.porter import *
import nltk
#nltk.download('wordnet')
from pprint import pprint
import numpy as np


np.random.seed(2018)
data = pd.read_csv('/Users/memoriessrls/Desktop/ML_Albs/Datasets/abcnews-date-text.csv', error_bad_lines=False)
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

print(len(documents))
#print(documents[:5])
st = PorterStemmer()


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def lemmatize_stemming(text):
    return st.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

doc_sample = documents[documents['index'] == 4].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
processed_docs = documents['headline_text'].map(preprocess)
processed_docs[:30]

#Create a dictionary from ‘processed_docs’ containing the number of times a word appears in the training set.
dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:                                     
        break

#Filter out tokens that appear in less than 15 docs, more than 0.5 docs, keeping the first 100000 most frequent tokens
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

# For each document we create a dictionary reporting how many words and how many times those words appear.
# Save this to ‘bow_corpus’, then check our selected document earlier.
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
bow_corpus[4]

bow_doc_4 = bow_corpus[4]
for i in range(len(bow_doc_4)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4[i][0], dictionary[bow_doc_4[i][0]], bow_doc_4[i][1]))
    #output del tipo: word 76("bushfir") appears 1 time


# tf-idf tecnica di info retrieval che pesa con uno score ogni termine o parola misurandone l'importanza rispetto ad un doc,
# in questo caso una sola riga del dataset, o rispetto ad una collezione di documenti
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
for doc in corpus_tfidf:
    pprint(doc)
    break

#adesso lda DIRICHLET LATENCY MODEL USANDO LA BAG OF WORDS
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
#per ogni topic esploreremo le parole che occorrono nel topic e il loro peso
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


#LDA USANDO TF-IDF
#output del tipo: topic 0, words: 0.035"govern" + 0.024"open" + 0.018"coast" e così via
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
#FONDAMENTALE-------------------------------------------------------------------------------------
#ho quindi creato un modello con 10 topics a partire dal bow corpus dato dall'insieme di documenti nel csv
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))   #STESSA COSA MA CON TF-IDF

#VALUTIAMO ADESSO A CHE TOPIC POSSA APPARTENERE IL NOSTRO DOCUMENTO AD ESEMPIO, IL NUMERO 4 DEL CSV
processed_docs[4]

for index, score in sorted(lda_model[bow_corpus[4]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


#POSSO USARE ANCHE UN DOCUMENTO NON APPARTENENTE ALLA COLLEZIONE PRESENTE NEL CSV
unseen_document = 'How a Pentagon deal became an identity crisis for Google'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
    
