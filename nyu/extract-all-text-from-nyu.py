import string 

from awesome_print import ap 
from nltk.tokenize import word_tokenize


punkt = set(string.punctuation)

def has_only_text(line):
	return all([token not in punkt and not token.isdigit() for token in word_tokenize(line)])

data  = [' '.join(word_tokenize(line)) for line in open('./biology-combined','rb').read().splitlines() 
		if not line == ''  and len(word_tokenize(line))>5 and has_only_text(line)]

with open('nyu-biology-cleaned-combined','wb') as outfile:
	for line in data:
		print>>outfile,line