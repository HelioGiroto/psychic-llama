import numpy as np 
def jaccard_similarity(a,b):
	a = set(a)
	b = set(b)

	try:
		return len(a & b)/float(len(a | b))
	except: 
		return 0 

def unique_words(aStr):
	return ' '.join([word for word in set(aStr.split())])

def princomp(A,numpc=3):
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
	[latent,coeff] = np.linalg.eig(np.cov(M))
	p = np.size(coeff,axis=1)
	idx = np.argsort(latent) # sorting the eigenvalues
	idx = idx[::-1]       # in ascending order
	# sorting eigenvectors according to the sorted eigenvalues
	coeff = coeff[:,idx]
	latent = latent[idx] # sorting eigenvalues
	if numpc < p and numpc >= 0:
		coeff = coeff[:,range(numpc)] # cutting some PCs if needed
	score = np.dot(coeff.T,M) # projection of the data in the new space
	return coeff,score,latent
