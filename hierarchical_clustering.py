import sys
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt

dataFile = sys.argv[1]
targetSize = int(sys.argv[2])

centroidMap = {}
clusterSizeMap = {}

centroidLine = False
clusterSizeLine = False

vectorLength = None

label = 0

with open(dataFile, 'r') as f:
	for line in f:
		if centroidLine:
			coords = map(float, line.split())
			if vectorLength == None:
				vectorLength = len(coords)
			centroidMap[label] = np.array(coords)
			centroidLine = False
			clusterSizeLine = True
		elif clusterSizeLine:
			clusterSize = map(int, line.split())[0]
			clusterSizeMap[label] = clusterSize
			centroidLine = True
			clusterSizeLine = False
			label += 1
		else:
			centroidLine = True

def closestLabels(centroidMap):
	closestLabels = None
	shortestDistance = None
	for label_i in centroidMap:
		centroid_i = centroidMap[label_i]
		for label_j in centroidMap:
			centroid_j = centroidMap[label_j]
			if label_i != label_j:
				distance_ij = LA.norm(centroid_i - centroid_j)
				if shortestDistance == None or distance_ij < shortestDistance:
					shortestDistance = distance_ij
					closestLabels = (label_i, label_j)
	return closestLabels

iterCount = 1

while len(centroidMap) > targetSize:
	if iterCount % 10 == 0:
		print "Iteration", iterCount
	iterCount += 1
	(i, j) = closestLabels(centroidMap)
	newLabel = min(i, j)
	n_i = clusterSizeMap[i]
	n_j = clusterSizeMap[j]
	C_i = centroidMap[i]
	C_j = centroidMap[j]
	newClusterSize = n_i + n_j
	newCentroid = (n_i * C_i + n_j * C_j) / (n_i + n_j)
	centroidMap[i] = newCentroid
	clusterSizeMap[i] = newClusterSize
	del centroidMap[j]
	del clusterSizeMap[j]

outputFileName = "hierarchical_centers.txt"

with open(outputFileName, 'w') as f:
	f.write(str(len(centroidMap)) + '\n')
	for label in centroidMap:
	   	# center for the cluster
	   	for i in range(vectorLength):
	   		if i < vectorLength - 1:
	   			f.write(str(centroidMap[label][i]) + ' ')
	   		else:
	   			f.write(str(centroidMap[label][i]) + '\n')
	   		# cluster size
	   	f.write(str(clusterSizeMap[label]) + '\n')
		



