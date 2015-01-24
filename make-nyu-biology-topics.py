import json, os, string,itertools

from nltk.tokenize import word_tokenize
from awesome_print import ap


stopwords = open('stopwords.txt','rb').read().splitlines()
punkt = set(string.punctuation)
NYU_BIOLOGY = './nyu/biology'
READ = 'rb'

def cleanse(word):
	return ''.join(ch for ch in word if ord(ch)<128)

def sanitize(list_of_text):
	words = itertools.chain.from_iterable([map(cleanse,word.lower().split()) for word in list_of_text if word != '' 
			and not any([urlword in  word for urlword in ['file://','mail','http']])
			and not all([ch in punkt or ch.isdigit() for ch in word.replace(" ","")])])
	return list(words)

nyu_data = {professor:sanitize(open(os.path.join(NYU_BIOLOGY,professor)).read().splitlines()) 
				for professor in os.listdir(NYU_BIOLOGY)}
json.dump(nyu_data,open('nyu-topics.json','wb'))