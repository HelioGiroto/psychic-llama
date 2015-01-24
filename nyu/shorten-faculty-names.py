longer_names = [''.join(line.split()) if len(line.split())<=2 else ''.join([line.split()[0], line.split()[-1]]) 
		for line in open('nyu-labs-cleaned-pruned').read().splitlines()] 

with open('shorter-names','wb') as outfile:
	for name in longer_names:
		print>>outfile,name