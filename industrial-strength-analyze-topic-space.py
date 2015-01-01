import os,nltk,json, itertools

import numpy as np 
import matplotlib.pyplot as plt
import Graphics as artist

from scipy.sparse.linalg import svds
from scipy.stats import scoreatpercentile
from awesome_print import ap
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import rcParams

rcParams['text.usetex'] = True

def to_ascii(aStr):
	return ''.join([ch for ch in aStr if ord(ch)<128])

def process(aStr):
	tagged = filter(lambda entry: entry[1] != None,map(to_wordnet_tags,nltk.pos_tag(word_tokenize(to_ascii(aStr)))))
	lemmatized = [lmtzr.lemmatize(glyph,pos=pos) for glyph,pos in tagged]
	return lemmatized

def to_wordnet_tags(tag_tuple):
	glyp,pos = tag_tuple
	if pos.startswith('J'):
		wordnet_pos = wordnet.ADJ
	elif pos.startswith('V'):
		wordnet_pos = wordnet.VERB
	elif pos.startswith('N'):
		wordnet_pos = wordnet.NOUN
	elif pos.startswith('R'):
		wordnet_pos = wordnet.ADV
	else:
		wordnet_pos = None #This ignore numbers, punctucation
	return (glyp,wordnet_pos)

avoid = ['combined']
directory = 'sinai'
lmtzr = WordNetLemmatizer()

'''
text = {filename: process(open(os.path.join(os.getcwd(),directory,filename),'rb').read())
				for filename in os.listdir(os.path.join(os.getcwd(),directory)) if 'combined' not in filename}

json.dump(text,open('./sinai/cleaned-tagged-text','wb'))
'''
text = json.load(open('./sinai/cleaned-tagged-text','rb'))
stopwords = open('stopwords.txt').read()
inputs = [' '.join(entry) for entry in text.values() if len(entry)>0]
tfx = TfidfVectorizer(inputs,tokenizer=word_tokenize,strip_accents='unicode',
	ngram_range=(1,3),min_df=3, use_idf=True)
tfidf = tfx.fit_transform(inputs)


#SVD of TFIDF Matrix
u,sigma,vt = svds(tfidf)

threshold_for_significant_sigma = np.percentile(sigma,25) #Cut off lowest quartile
sigma = [num if num > threshold_for_significant_sigma else 0 for num in sigma ]
sigma = np.diagflat(sigma) #Prior steps preserves original dimensions
reconstructed_matrix =u.dot(sigma).dot(vt)
print reconstructed_matrix.shape

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(reconstructed_matrix[:,0],reconstructed_matrix[:,1],'k.')
artist.adjust_spines(ax)
ax.set_xlabel(r'\Large $1^{st}$ \textbf{\textsc{Semantic Dimension}')
ax.set_ylabel(r'\Large $2^{nd}$ \textbf{\textsc{Semantic Dimension}')
plt.tight_layout()
plt.savefig('LSA-quick.png',dpi=300)