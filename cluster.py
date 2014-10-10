from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
import math
from numpy import linalg as LA

def dataWithLabel(data, label, labels):
	indices = []
	for i in range(0, len(data)):
		if labels[i] == label:
			indices.append(i)
	return data[indices]

def initialCentroids(clusterCenters, data, labels):
	centroidsMap = {}
	for i in range(len(clusterCenters)):
		C_i = clusterCenters[i]
		S_i = dataWithLabel(data, i, labels)
		centroidsMap[i] = Cluster(i, C_i, S_i)
	return centroidsMap

class Cluster:
	def __init__(self, label, centroid, points):
		self.label = label
		self.centroid = centroid
		self.points = points

	def isViolation(self, theta):
		for point in self.points:
			if LA.norm(point - self.centroid) > math.sqrt(theta) * LA.norm(self.centroid):
				return True
		return False

class ClusterSet:
	def __init__(self, data):
		self.data = data
		self.clusterMap = None

	def normalize(self):
		for i in range(len(self.data)):
			consumption = sum(self.data[i])
			for j in range(len(self.data[i])):
				self.data[i][j] /= consumption

	def maxLabel(self):
		maxLabel = None
		for label in self.clusterMap.keys():
			if (maxLabel == None or int(label) > maxLabel):
				maxLabel = int(label)
		return maxLabel

	def fitData(self, K):
		km = None
		if not self.clusterMap:
			km = KMeans(n_clusters=K, init='k-means++', max_iter=100, n_init=1)
		else:
		  	# use centroids from last iteration
		  	centroids = self.getCentroids()
		  	km = KMeans(n_clusters=K, init=centroids, max_iter=100, n_init=1)
		km.fit(self.data)
		self.clusterMap = initialCentroids(km.cluster_centers_, self.data, km.labels_)

	def getCentroids(self):
		centroids = []
		for cluster in self.clusterMap.values():
			centroids.append(cluster.centroid)
		return np.array(centroids)

	def getCluster(self, label):
		return self.clusterMap[label]

	def findViolations(self, theta):
		indexes = []
		for cluster in self.clusterMap.values():
			if cluster.isViolation(theta):
				indexes.append(cluster.label)
		return indexes

	def splitLabel(self, label):
		km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
		cluster = self.getCluster(label)

		km.fit(cluster.points)
		labelOffset = self.maxLabel() + 1

		S_0 = dataWithLabel(cluster.points, 0, km.labels_)
		C_0 = km.cluster_centers_[0]
		cluster0 = Cluster(labelOffset, C_0, S_0)
		self.clusterMap[labelOffset] = cluster0



		S_1 = dataWithLabel(cluster.points, 1, km.labels_)
		C_1 = km.cluster_centers_[1]
		cluster1 = Cluster(labelOffset + 1, C_1, S_1)
		self.clusterMap[labelOffset] = cluster1

		del self.clusterMap[label]

	def smallestCluster(self):
		smallest = None
		for label in self.clusterMap:
			if smallest == None or \
			   len(self.clusterMap[label].points) < len(self.clusterMap[smallest].points):
				smallest = label
		return smallest





