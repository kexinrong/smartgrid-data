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

thetaValues = .0001 * np.arange(1, 10)
clusterSizes = []

for theta in thetaValues:
	print "Theta:\n", theta
	K = minK
	# Initial cluster set
	clusterSet = ClusterSet(data)
	clusterSet.normalize()

	while True:
		clusterSet.fitData(K)
		n_v = clusterSet.findViolations(theta)
		K += len(n_v)

		for label in n_v:
			clusterSet.splitLabel(label)

		if len(n_v) == 0:
			clusterSizes.append(K)
			break

plt.scatter(thetaValues, clusterSizes)
plt.title("Number of clusters as a function of $\\theta$")
plt.xlabel("$\\theta$")
plt.ylabel("Number of clusters")
plt.xlim(0, 1.1 * max(thetaValues))
plt.ylim(0, 1.1 * max(clusterSizes))
plt.show()

plt.scatter(thetaValues, np.log(clusterSizes))
plt.title("log(Number of clusters) as a function of $\\theta$")
plt.xlabel("$\\theta$")
plt.ylabel("log(Number of clusters)")
plt.xlim(0, 1.1 * max(thetaValues))
plt.ylim(0, 1.1 * max(np.log(clusterSizes)))
plt.show()