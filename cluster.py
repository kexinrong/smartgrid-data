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


def initialCentroids(clusterCenters):
	centroidsMap = {}
	for i in range(len(clusterCenters)):
		centroidsMap[str(i)] = clusterCenters[i]
	return centroidsMap

def dataWithLabel(l, labels, data):
	indices = []
	for i in range(0, len(labels)):
		if labels[i] == l:
			indices.append(i)
	S_l = data[indices]
	return (indices, S_l)

def thresholdTest(S, centroid, theta):
	totalDistance = 0
	for x in S:
		totalDistance +=  LA.norm(x - centroid)
	threshold = theta * LA.norm(centroid)
	print totalDistance
	return totalDistance < threshold

def maxLabel(centroidMap):
	maxLabel = None
	for label in centroidsMap:
		if (currentMax == None or int(label) > currentMax):
			currentMax = int(label)
	return maxLabel

def updateLabels(S, indices, l, centroidMap, labels):
	km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
	km.fit(S)
	labelOffset = maxLabel(centroidMap) + 1
	print km.labels_

centroidMap = initialCentroids(km.cluster_centers_)
labels = km.labels_

(indices, S_0) = dataWithLabel(0, km.labels_, data)
updateLabels(S_0, indices, 0, centroidMap, labels)
print thresholdTest(S_0, km.cluster_centers_, theta)


