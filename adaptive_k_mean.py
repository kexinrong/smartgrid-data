import sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt

from cluster import Cluster, ClusterSet


fileName = sys.argv[1]
minK = int(sys.argv[2])
theta = float(sys.argv[3])

vectorLength = None

with open(fileName, 'r') as dataFile:
  firstLine = dataFile.readline()
  vectorLength = int(firstLine)

data = np.loadtxt(fileName,
	skiprows = 1,
	usecols = range(1, vectorLength + 1))

ids = np.loadtxt(fileName,
	skiprows = 1,
	usecols = [0])

def plot_cluster(cluster):
	for shape in cluster.points:
		plt.plot(shape, color='black')
	plt.plot(cluster.centroid, 'o', markerfacecolor='None', 
		     markeredgewidth=2, markeredgecolor='red')
	plt.xlabel('Hour')
	plt.ylabel('Normal Usage')
	plt.title('#' + str(cluster.label))
	plt.show()

K = minK
# Initial centroid
centroids = np.zeros([K, vectorLength], dtype=np.float)


clusterSet = ClusterSet(data)
clusterSet.normalize()

while True:
	clusterSet.fitData(K)
	n_v = clusterSet.findViolations(theta)
	K += len(n_v)

	for label in n_v:
		clusterSet.splitLabel(label)

	if len(n_v) == 0:
		for cluster in clusterSet.clusterMap.values():
			print len(cluster.points)
		# plot the smallest cluster
		l = clusterSet.largestCluster()
		print "Total clusters: ", K
		print "Smallest cluster size: ", len(clusterSet.getCluster(l).points)
		plot_cluster(clusterSet.getCluster(l))

		f = open("adaptive_k_centers.txt", "w")
		# total number of clusters
		f.write(str(K) + '\n')
		for label in clusterSet.clusterMap.keys():
			# center for the cluster
			cluster = clusterSet.getCluster(label)
			centroid = cluster.centroid
			for i in range(vectorLength):
				if i < vectorLength - 1:
					f.write(str(centroid[i]) + ' ')
				else:
					f.write(str(centroid[i]) + '\n')
			# cluster size
			f.write(str(len(cluster.points)))
		f.close()
		break
