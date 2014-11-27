from collections import deque
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

ROOT = 1 # Root node
HOUSE_START = 525 # start index of house
HOUSE_END = 1933 # end index of hosue
N = 2065 # Number of nodes

# Build tree
tree = {}
for i in range(1, N + 1):
	tree[i] = []
lines = open('lines.txt', 'r')
for i in range(N - 1):
    line = lines.readline().split()
    tree[int(line[0])].append(int(line[1]))
lines.close()

m = loadmat('../data/output_bfs_social/2012_07_15_00_00_00_PDT/case2065_BFS_2012baseline_2012_07_15_00_00_00_PDT_20140325_1424.mat')
branch = m['network']['branch'][0][0]
r = {} # resistance
for i in range(N - 1):
    x = int(branch[i][0].real)
    y = int(branch[i][1].real)
    idx = 5 # A
    if int(branch[i][3].real) == 1:
        idx = 9 # B
    elif int(branch[i][4].real) == 1:
        idx = 13 # C
    r[(x, y)] = branch[i][idx].real


queue = deque()
queue.append(ROOT)
hops = {}
res = {}
hops[ROOT] = 0
res[ROOT] = 0
while len(queue) > 0:
    curr = queue.popleft()
    for node in tree[curr]:
        hops[node] = hops[curr] + 1
        res[node] = res[curr] + r[(curr, node)]
        queue.append(node)


# Plot histogram 
x = []
for i in range(HOUSE_START, HOUSE_END + 1):
	x.append(hops[i])
bin = max(x) - min(x) + 1
weights = np.ones_like(x)/float(len(x))
plt.hist(x, bin, weights = weights, color='green', alpha = 0.5)
plt.title("Histogram of household's hops from substation")
plt.xlabel("# hops from substation")
plt.ylabel("Frequency")
plt.show()


# Plot histogram 
x = []
for i in range(HOUSE_START, HOUSE_END + 1):
	x.append(res[i])
weights = np.ones_like(x)/float(len(x))
plt.hist(x, bin, weights = weights, color='green', alpha = 0.5)
plt.title("Histogram of household's total resistance ")
plt.xlabel("Total Resistance")
plt.ylabel("Frequency")
plt.show()

