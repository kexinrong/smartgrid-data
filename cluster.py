import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten

dataFile = "data/case2065_BFS_2012baseline_2012_07_15_00_00_00_PDT_20140325_1424.txt"
initialK = 100

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

clusters = kmeans(data, initialK)

clusters