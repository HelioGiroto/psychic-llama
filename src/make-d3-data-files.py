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

#Circuitous code because starting and ending points need to be dictionary with keys in the same order. 
data = dict(sinai.items() + nyu.items())
labs = data.keys()
basis_vectors = [set(line.split()) for line in open('topics-as-sets.txt').read().splitlines()]

projection_data_onto_basis_vectors = np.array([[tech.jaccard_similarity(data[lab],basis_vector) 
											for basis_vector in basis_vectors] 
											for lab in labs])


eigenvectors,projections,eigenvalues = tech.princomp(projection_data_onto_basis_vectors,numpc=3)

#NVD3 excpets the format of {Group:[{x:,y:,size:,shape:}]}
#Orthogonalize basisvectors

shapes = {'nyu':'circle','sinai':'cross'}

sinai_data = []
nyu_data = []
for i,projection in enumerate(projections.T):
	x,y,size = np.absolute(projection)
	university = 'nyu' if labs[i] in nyu else 'sinai'
	shape = shapes[university]
	if university == 'nyu':
		nyu_data.append({'x':x,'y':y,'size':size,'shape':shape})
	elif university == 'sinai':
		sinai_data.append({'x':x,'y':y,'size':size,'shape':shape,'label':labs[i]})
	else:
		ap('Univeristy not recognized')
		pass

json.dump([{'key':'NYU','values':nyu_data},{'key':'Sinai','values':sinai_data}],open('data.json','wb'))

'''
for university,color,name in zip([nyu,sinai],['r','k'],['nyu','sinai']):
	keys  = university.keys()	
	with open('%s-projections.csv'%name,'wb') as outfile:
		for projection,pi in zip(projections.T,keys):
			projection = np.absolute(projection)
			print>>outfile,'%.04f,%.04f,%.04f,%.04f,%s,%s'%(projection[0],projection[1],projection[2],projection[3],name,pi)
	np.savetxt('%s-projections-noname.csv'%name, np.absolute(projections).T,delimiter=',',fmt='%.04f')
'''