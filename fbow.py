# import gzip
import gensim
import math
import pprint

from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
from gensim import similarities


#corpus = api.load('20-newsgroups')  # download the corpus and return it opened as an iterable
#sentences = []

#for line in corpus:
#    sentences.append(line['data'].split(' '))

#print(sentences[12])
    
model = Word2Vec.load("20-newsgroups.model")#Word2Vec(sentences)  # train a model from the corpus

wordcount = model.wv.vocab['data'].count

fbow = []

for i in range(wordcount):
    tmp = []
    for j in range(wordcount):
        tmp.append(model.wv.similarity(model.wv.index2word[i],model.wv.index2word[j]))
    fbow.append(tmp)
    
pprint.pprint(fbow[0:5])   
#print("Cosine similarity between 'man' embedding and 'woman' embedding is: " + str(model.wv.similarity("man", "woman")))
#print("Cosine similarity between 'man' embedding and 'insect' embedding is: " + str(model.wv.similarity("man", "insect")))
 
