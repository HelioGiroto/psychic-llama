import json

import utils as tech 
import numpy as np 
import matplotlib.pyplot as plt 
import Graphics as artist
import matplotlib as mpl 

from awesome_print import ap 
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
rcParams['text.usetex'] = True

TEXT = 1
sinai = {key:set(val) for key, val in json.load(open('sinai-topics.json','rb')).iteritems()}
nyu = {key:set(val) for key, val in json.load(open('nyu-topics.json','rb')).iteritems()}
data = dict(sinai.items() + nyu.items())

basis_vectors = [set(line.split()) for line in open('topics-as-sets.txt').read().splitlines()]

projection_data_onto_basis_vectors = np.array([[tech.jaccard_similarity(data[lab],basis_vector) 
											for basis_vector in basis_vectors] 
											for lab in data])


basis_vector_correlation = np.array([[tech.jaccard_similarity(one,two) for one in basis_vectors] 
											for two in basis_vectors])


#Orthogonalize basisvectors


fig = plt.figure()
ax = fig.add_subplot(111)
plt.hold(True)
for university,color,name in zip([nyu,sinai],['r','k'],['nyu','sinai']):
	keys  = university.keys()
	projection_data_onto_basis_vectors = np.array([[tech.jaccard_similarity(data[lab],basis_vector) 
											for basis_vector in basis_vectors] 
											for lab in keys])

	eigenvectors,projections,eigenvalues = tech.princomp(projection_data_onto_basis_vectors,numpc=4)

	
	with open('%s-projections.csv'%name,'wb') as outfile:
		for projection,pi in zip(projections.T,keys):
			projection = np.absolute(projection)
			print>>outfile,'%.04f,%.04f,%.04f,%.04f,%s,%s'%(projection[0],projection[1],projection[2],projection[3],name,pi)

	np.savetxt('%s-projections-noname.csv'%name, np.absolute(projections).T,delimiter=',',fmt='%.04f')
	ax.scatter(np.absolute(projections[1,:]),np.absolute(projections[2,:]), 
	c=color, cmap=mpl.cm.gray)

ax.set_xlabel(artist.format('Second Semantic Dimension'))
ax.set_ylabel(artist.format('Third Semantic Dimension'))
plt.tight_layout()
artist.adjust_spines(ax)
plt.savefig('semantic-space-combined-colored.png',dpi=300)

