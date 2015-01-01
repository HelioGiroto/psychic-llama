import os,nltk,json, itertools

from awesome_print import ap
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

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
stopwords = open('stopwords.txt').read().splitlines()
text = json.load(open('./sinai/cleaned-tagged-text','rb'))
#Create the vectors
vector_basis = set(itertools.chain.from_iterable(text.values()))
print vector_basis