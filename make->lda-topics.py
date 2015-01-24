import gensim, json,os

import numpy as np

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from awesome_print import ap
from gensim import corpora, models, similarities

if not os.path.isfile('./combined.mm'):
	data = json.load(open('sinai-topics.json','rb')).values() + json.load(open('nyu-topics.json','rb')).values()
	all_tokens = sum(data,[])
	tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word)==1)
	texts = [[word for word in line if word not in tokens_once] for line in data]

	dictionary = corpora.Dictionary(texts)
	dictionary.save('./combined.dict')

	corpus = [dictionary.doc2bow(text) for text in texts]
	corpora.MmCorpus.serialize('./combined.mm',corpus)
else:
	dictionary = corpora.Dictionary.load('./combined.dict')
	corpus = corpora.MmCorpus('./combined.mm')


lda = models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,passes=20)
with open('./topics-combined.txt','wb') as outfile:
	for topic in lda.print_topics(num_topics=100,num_words=40):
		print>>outfile,topic

'''
tfx = TfidfVectorizer(data,tokenizer=word_tokenize,strip_accents='unicode',
	ngram_range=(1,3),min_df=3, use_idf=True)

'''

'''
topic_word = model.topic_word_
n_top_words = 10

ap(model.doc_topic_[0])

''''''
with open('lda-topics.txt','wb') as outfile:
	for i,topic_dist in enumerate(topic_word):
		topic_words = tfidf[i,np.argsort(topic_dist)][:-n_top_words:-1]

		print>>outfile,'Topic {}: {}'.format(i, ' '.join(topic_words))
'''