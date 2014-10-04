import sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy import linalg as LA

dataFile = sys.argv[1]
minK = int(sys.argv[2])
theta = float(sys.argv[3])

vectorLength = None

with open(dataFile, 'r') as f:
  firstLine = f.readline()
  vectorLength = int(firstLine)

data = np.loadtxt(dataFile,
	skiprows = 1,
	usecols = range(1, vectorLength + 1))

ids = np.loadtxt(dataFile,
	skiprows = 1,
	usecols = [0])

km = KMeans(n_clusters=minK, init='k-means++', max_iter=100, n_init=1)
clusters = km.fit(data)

def dataWithLabel(l, labels, data):
	indices = []
	for i in range(0, len(labels)):
		if labels[i] == l:
			indices.append(i)
	S_l = []
	for index in indices:
		S_l.append(data[index])
	return S_l

def thresholdTest(S, centroid, theta):
	totalDistance = 0
	for x in S:
		totalDistance +=  LA.norm(x - centroid)
	threshold = theta * LA.norm(centroid)
	print totalDistance
	return totalDistance < threshold



S_0 = dataWithLabel(0, km.labels_, data)
print S_0
print thresholdTest(S_0, km.cluster_centers_, theta)


