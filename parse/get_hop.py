from collections import deque
import matplotlib.pyplot as plt

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

dist = 0
queue = deque()
queue.append(ROOT)
hops = {}
hops[ROOT] = 0
while len(queue) > 0:
	curr = queue.popleft()
	for node in tree[curr]:
		hops[node] = hops[curr] + 1
		queue.append(node)


# Plot histogram 
sizes = []
for i in range(HOUSE_START, HOUSE_END + 1):
	sizes.append(hops[i])
print sizes
plt.hist(sizes, max(sizes) - min(sizes) + 1, normed = 1, color='green', alpha = 0.5)
plt.title("Histogram of households' hops from substation")
plt.xlabel("# hops from substation")
plt.ylabel("Frequency")
plt.show()

