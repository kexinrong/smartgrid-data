import sys, math
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from numpy import linalg as LA
import matplotlib.pyplot as plt
import operator

fileName = sys.argv[1]
numberOfLoadShapes = int(sys.argv[2])

centroidMap = {}
clusterSizeMap = {}

centroidLine = False
clusterSizeLine = False

vectorLength = None

label = 0

totalClusters = None

with open(fileName, 'r') as f:
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
			totalClusters = int(line)
			centroidLine = True

sortedClusterMapEntries = sorted(clusterSizeMap.items(), key=operator.itemgetter(1), reverse=True)
sortedClusterSizeLabels = map(lambda x: x[0], sortedClusterMapEntries)
mostCommonClusterSizeLabels = sortedClusterSizeLabels[:numberOfLoadShapes]

x = range(1, 25)

w = math.ceil((numberOfLoadShapes) ** .5)
h = math.ceil((numberOfLoadShapes) ** .5)

for i in range(1, numberOfLoadShapes + 1):
	label = mostCommonClusterSizeLabels[i - 1]
	y = centroidMap[label]
	plt.subplot(h, w, i)
	plt.scatter(x, y)
	clusterSize = mostCommonClusterSizeLabels[i - 1]

	percent = 100. * clusterSize / totalClusters
	plt.title(str(percent) + "%")
	plt.xlabel("Hour")
	plt.ylabel("Norm. Usage")
	plt.xlim(0, 25)

plt.tight_layout()
plt.show()

