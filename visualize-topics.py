import numpy as np
import Graphics as artist
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import matplotlib.cm as cmx 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams

rcParams['text.usetex'] = True 

filenames = ['basis-vector-correlation-matrix-reduced','data-correlation-matrix-reduced']

data = {filename:np.loadtxt(filename) for filename in filenames}

jet = plt.get_cmap('jet')

for name,datum in data.iteritems():
	fig = plt.figure()
	ax = fig.add_subplot(111,projection='3d')
	#cNorm = colors.Normalize(vmin=datum[:,2].min(),vmax=datum[:,2].max())
	#scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
	#this_plots_colors = [scalarMap.to_rgba(point) for point in datum[:,2]]
	#cax = ax.scatter(datum[:,0],datum[:,1],datum[:,2],marker='.',cmap=jet,c=datum[:,2],edgecolor=None,s=30)
	ax.scatter(datum[:,0],datum[:,1],datum[:,2],marker='.',color='k')
	#artist.adjust_spines(ax)
	ax.set_xlabel(artist.format('First Semantic Dimension'))
	ax.set_ylabel(artist.format('Second Semantic Dimension'))
	#cbar = plt.colorbar(cax)
	ax.set_zlabel(artist.format('Third Semantic Dimension'))
	plt.tight_layout()
	plt.show()