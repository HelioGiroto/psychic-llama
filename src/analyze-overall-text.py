import nltk, itertools,string
import matplotlib.pyplot as plt 
import Graphics as artist

from nltk.tokenize import word_tokenize
from matplotlib import rcParams

rcParams['text.usetex'] = True

def to_ascii(iterable):
	return ''.join([item for item in iterable if ord(item)<128])

stopwords = open('stopwords.txt','r').read().splitlines()
data = itertools.chain.from_iterable([word_tokenize(to_ascii(line)) for line in open('./sinai/cleaned-combined','rb').read().splitlines()])
data = [word for word in data if word not in set(string.punctuation) and word not in stopwords]

fdist =  nltk.FreqDist(data)
words,freqs = zip(*fdist.most_common(25))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(freqs,'k',linewidth=2)
ax.set_xticks(range(len(freqs)))
ax.set_xticklabels(map(artist.format,words),rotation='vertical')
ax.set_title(artist.format('Most common words describing Sinai researchers'))
ax.set_ylabel(artist.format('No. of times word appeared'))
plt.tight_layout()
plt.savefig('overall-sinai-frequencies.png')