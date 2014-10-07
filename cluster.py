import sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt


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


def initialCentroids(clusterCenters):
	centroidsMap = {}
	for i in range(len(clusterCenters)):
		centroidsMap[i] = clusterCenters[i]
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
	return totalDistance < threshold

def maxLabel(centroidMap):
	maxLabel = None
	for label in centroidMap:
		if (maxLabel == None or int(label) > maxLabel):
			maxLabel = int(label)
	return maxLabel

def updateLabels(S, indices, centroidMap):
	km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
	km.fit(S)
	labelOffset = maxLabel(centroidMap) + 1
	return km.cluster_centers_

def plot_cluster(label, centroidMap, S):
	for shape in S:
		plt.plot(shape, color='black')
	plt.plot(centroidMap, 'o', markerfacecolor='None', 
		     markeredgewidth=2, markeredgecolor='red')
	plt.xlabel('Hour')
	plt.ylabel('Normal Usage')
	plt.title('#' + str(label))
	plt.show()

centroidMap = {}
K = minK
# Initial centroid
centroids = np.zeros([K, vectorLength], dtype=np.float)

while True:
	# Run kmeans
	if not centroidMap:
		km = KMeans(n_clusters=K, init='k-means++', max_iter=100, n_init=1)
	else:
		km = KMeans(n_clusters=K, init=centroids, max_iter=100, n_init=1)
	km.fit(data)
	centroidMap = initialCentroids(km.cluster_centers_)
	centroids = km.cluster_centers_

	n_v = []
	S = []
	for label in centroidMap:
		S.append([])
	for label in centroidMap:
		(indices, S[label]) = dataWithLabel(label, km.labels_, data)
		# Record clusters violating the threshold test
		if not thresholdTest(S[label], centroidMap[label], theta):
			n_v.append(centroidMap[label])
			# Run K-mean with K = 2
			new_centers = updateLabels(S[label], indices, centroidMap)
			# Add new centroids
			centroids = np.append(centroids, new_centers, axis = 0)

	if len(n_v) == 0:
		# Plot the biggest cluster
		biggest = 0
		for label in centroidMap:
			if len(S[label]) > len(S[biggest]):
				biggest = label
		print "Total clusters: ", K
		print "Biggest cluster size: ", len(S[biggest]) 
		l = biggest # label to plot
		plot_cluster(l, centroidMap[l], S[l])

		# Output centroids to file
		f = open("adaptive_k_centers.txt", "w")
		f.write(str(K) + '\n')
		for label in centroidMap:
			for i in range(vectorLength):
				if i < vectorLength - 1:
					f.write(str(centroidMap[label][i]) + ' ')
				else:
					f.write(str(centroidMap[label][i]) + '\n')
		f.close()
		break
	else:
		K += 2 * len(n_v)

