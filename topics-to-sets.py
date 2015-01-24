#data = [[token.split('*')[1] for token in line.split(' + ')] for line in open('./topics-combined.txt').read().splitlines()]
data = [[token.split('*')[-1] for token in line.split(' + ')] for line in open('./topics-combined.txt','rb').read().splitlines()]
with open('combined-topics-as-sets.txt','wb') as outfile:
	for line in data:
		print>>outfile,' '.join(line)
