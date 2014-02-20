

def shortest(dists):
	return min(dists)

def longest(dists):
	return max(dists)

def average(dists):
	return sum(dists) / float(len(dists))

class HAC:
	
	def __init__(self, objs, pair_dist, cluster_dist, as_idx=True):
		self.objs = objs
		self.pair_dist = pair_dist
		self.cluster_dist = cluster_dist
		self.as_idx = as_idx

	def init_clusters(self):
		self.clusters = []
		for x in xrange(len(self.objs)):
			self.clusters.append( (x,) )

	def pair_dists(self, clust1, clust2):
		dists = []
		for idx1 in clust1:
			for idx2 in clust2:
				dist = self.pair_dist(self.objs[idx1], self.objs[idx2])
				dists.append(dist)
		return dists

	def cluster(self, k_range=None):
		if k_range is None:
			k_range = [-1, -1]
		self.init_clusters()
		cluster_sets = dict()
		for k in reversed(xrange(1, len(self.objs))):
			if k_range[0] <= k <= k_range[1]:
				cluster_set = self.iterate(True)
				cluster_sets[k] = cluster_set
			else:
				self.iterate(False)
			assert k == len(self.clusters)
		return cluster_sets

	def iterate(self, return_clusters=False):
		min_metric = 2.0 ** 20
		one = two = None
		for x, clust1 in enumerate(self.clusters):
			for y, clust2 in enumerate(self.clusters):
				if y == x:
					continue
				dists = self.pair_dists(clust1, clust2)
				metric = self.cluster_dist(dists)
				if metric < min_metric:
					min_metric = metric
					one = clust1
					two = clust2

		# combine clusters one and two
		#print "%f\t %s %s" % (min_metric, one, two)
		combined = one + two
		self.clusters.remove(one)
		self.clusters.remove(two)
		self.clusters.append(combined)
		if return_clusters:
			if self.as_idx:
				return list(self.clusters)
			else:
				return self.itemize(self.clusters)

	def itemize(self, cluster_set):
		new_cluster_set = []
		for cluster in cluster_set:
			new_cluster = tuple(map(lambda x: self.objs[x], cluster))
			new_cluster_set.append(new_cluster)
		return new_cluster_set
		
if __name__ == "__main__":
	data = [1, 5, 10, 11, 6, 2, 1, 0, -1, 15, 14]
	print "data len:", len(data)
	print data
	hac = HAC(data, lambda x, y: abs(x - y), average)
	cluster_sets = hac.cluster([1, len(data)])
	for key in reversed(sorted(cluster_sets.keys())):
		print key, cluster_sets[key]
		
