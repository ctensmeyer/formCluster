
import network
import utils
import doc

import multiprocessing
import collections
import itertools
import random

class Cluster:
	
	def __init__(self, members, center, _id = None):
		self.members = members
		self.center = center
		self.label = self.center.label
		self._id = _id

	def set_label(self):
		labels = map(lambda doc: doc.label, self.members)
		c = collections.Counter(labels)
		try:
			self.label = c.most_common(1)[0][0]
		except:
			self.label = None


class BaseCONFIRM(object):
	
	NEW_CLUSTER = -1
	def __init__(self, docs, sim_thresh=None, **kwargs):
		self.docs = docs
		self.clusters = list()
		self.num_clustered = 0
		self.sim_thresh = sim_thresh
		self._cached_doc = None

	def _before_iteration(self, _doc, **kwargs):
		pass

	def _after_iteration(self, _doc, **kwargs):
		if self.num_clustered % 10 == 0:
			print "%d documents processed" % self.num_clustered

	def _add_cluster(self, _doc, member=True):
		prototype = _doc.copy(len(self.clusters))
		prototype.label = None
		members = list()
		if member:
			members.append(_doc)
		cluster = Cluster(members, prototype)
		self.clusters.append(cluster)
		return cluster

	def _add_to_cluster(self, cluster, _doc):
		cluster.center.aggregate(_doc)
		cluster.members.append(_doc)

	def _sufficiently_similar(self, _doc, cluster, sim_score, **kwargs):
		return sim_score > self.sim_thresh

	def _calc_sim_scores(self, _doc):
		return map(lambda cluster: self.cluster_doc_similarity(cluster, _doc), self.clusters)

	def _get_cached_sim_scores(self, _doc):
		if not _doc is self._cached_doc:
			self._cached_sim_scores = self._calc_sim_scores(_doc)
			self._cached_doc = _doc
		return self._cached_sim_scores

	def _cluster_sim_scores(self, _doc):
		return self._get_cached_sim_scores(_doc)

	def _most_similar_cluster(self, _doc):
		similarities = self._cluster_sim_scores(_doc)
		idx = utils.argmax(similarities)
		cluster = self.clusters[idx]
		similarity = similarities[idx]
		return cluster, similarity

	def _choose_cluster(self, _doc):
		cluster, similarity = self._most_similar_cluster(_doc)
		self._cached_most_similar_val = similarity
		if self._sufficiently_similar(_doc, cluster, similarity):
			return cluster
		else:
			return self.NEW_CLUSTER

	def _init_clusters(self):
		pass

	def cluster(self):
		self._init_clusters()
		for x, _doc in enumerate(self.docs):
			self._before_iteration(_doc)
			_doc._load_check()
			new_cluster = False
			if not self.clusters:
				self._add_cluster(_doc)
			else:
				cluster = self._choose_cluster(_doc)
				if cluster == self.NEW_CLUSTER:
					self._add_cluster(_doc)
					new_cluster = True
				else:
					self._add_to_cluster(cluster, _doc)
			self.num_clustered += 1
			self._after_iteration(_doc, new_cluster=new_cluster)
		self.post_process_clusters()

	def post_process_clusters(self, **kwargs):
		for cluster in self.clusters:
			cluster.center.final_prune()	

	def doc_similarity(self, doc1, doc2):
		return doc1.similarity(doc2)

	def cluster_similarity(self, cluster1, cluster2):
		return self.doc_similarity(cluster1.center, cluster2.center)

	def cluster_doc_similarity(self, cluster, _doc):
		return self.doc_similarity(cluster.center, _doc)

	# Can be used before clustering to init some clusters
	def init_cluster(self, _doc):
		prototype = _doc.copy(len(self.clusters))
		prototype.label = None
		self.clusters.append(Cluster([], prototype))

	def get_clusters(self):
		return self.clusters

	def get_docs(self):
		return self.docs

	# Can be expensive if there are lots of clusters
	def get_cluster_sim_mat(self):
		mat = []
		for clust1 in self.clusters:
			row = []
			for clust2 in self.clusters:
				if clust1 == clust2:
					row.append(1.0)
				else:
					row.append(self.cluster_similarity(clust1, clust2))
			mat.append(row)
		return mat

	# not reccommended unless the data size is small
	def get_doc_sim_mat(self):
		mat = []
		for doc1 in self.docs:
			row = []
			for doc2 in self.docs:
				if doc1 == doc2:
					row.append(1.0)
				else:
					row.append(self.doc_similarity(doc1, doc2))
			mat.append(row)
		return mat

	# may be expensive
	def get_doc_cluster_sim_mat(self):
		mat = []
		for _doc in self.docs:
			row = []
			for cluster in self.clusters:
				row.append(self.cluster_doc_similarity(cluster, _doc))
			mat.append(row)
		return mat


class AnalysingCONFIRM(BaseCONFIRM):

	def _cluster_sim_scores(self, _doc):
		info = [(cluster, 
			[0, _doc.similarities_by_name(cluster.center).values(), cluster_idx, cluster.label]
			) for cluster_idx, cluster in enumerate(self.clusters)]
		for i in info:
			i[1][0] = utils.harmonic_mean_list(i[1][1])
		info.sort(key=lambda i: -1 * i[1][0])
		val = info[0][1][0]
		for i in info:
			i[1][0] = "%.3f" % i[1][0]
			i[1][1] = map(lambda num: "%.3f" % num, i[1][1])

		cluster_match = info[0][0]

		# print out stuff here
		toprint = "\t".join(map(str, 
			[self.num_clustered, _doc._id, _doc.label == cluster_match.label, _doc.label, cluster_match.label, len(self.clusters)]))
		for y, i in enumerate(info):
			if y > 2:
				break
			toprint += "\t" + str(i[1])
		print toprint
		return super(AnalysingCONFIRM, self)._cluster_sim_scores(_doc)

	def _after_iteration(self, _doc, **kwargs):
		pass

class PruningCONFIRM(BaseCONFIRM):
	
	def post_process_clusters(self, min_size=5, **kwargs):
		''' 
		Prune all clusters of size < minsize
		:return: list of docs that were members of the pruned clusters
		'''
		super(PruningCONFIRM, self).post_process_clusters(**kwargs)
		odd_docs = list()
		clusters_to_remove = list()
		for cluster in self.clusters:
			if len(cluster.members) < min_size:
				odd_docs += cluster.members
				clusters_to_remove.append(cluster)
		for cluster in clusters_to_remove:
			self.clusters.remove(cluster)
		return odd_docs

class IsolatePruningCONFIRM(PruningCONFIRM):
	
	def post_process_clusters(self, min_size=5, **kwargs):
		''' 
		Take all docs in clusters of size < minsize and stick them in a single isolated cluster
		'''
		odd_docs = super(IsolatePruningCONFIRM, self).post_process_clusters(min_size, **kwargs)
		if odd_docs:
			# make a single cluster of the oddballs
			odd_cluster = self._add_cluster(odd_docs[0])
			for _doc in odd_docs[1:]:
				odd_cluster.members.append(_doc)

class RedistributePruningCONFIRM(PruningCONFIRM):
	
	def post_process_clusters(self, min_size=5, **kwargs):
		''' 
		Take all docs in clusters of size < minsize and assign them to the most similar cluster
			of size >= minsize
		'''
		odd_docs = super(RedistributePruningCONFIRM, self).post_process_clusters(min_size, **kwargs)
		for _doc in odd_docs:
			cluster = self._most_similar_cluster(_doc)[0]
			cluster.members.append(_doc)

class TwoPassCONFIRM(BaseCONFIRM):
	
	def post_process_clusters(self, **kwargs):
		''' Reassign all docs to the most similar cluster.  Does not change prototypes '''
		super(TwoPassCONFIRM, self).post_process_clusters(**kwargs)

		# clear assignments
		for cluster in self.clusters:
			cluster.members = list()
		for _doc in self.docs:
			cluster = self._most_similar_cluster(_doc)[0]
			cluster.members.append(_doc)

class PerfectCONFIRM(BaseCONFIRM):
	
	def _add_cluster(self, _doc, member=True):
		prototype = _doc.copy(len(self.clusters))
		prototype.label = _doc.label
		members = list()
		if member:
			members.append(_doc)
		cluster = Cluster(members, prototype)
		self.clusters.append(cluster)
		return cluster

	def _choose_cluster(self, _doc):
		for cluster in self.clusters:
			if cluster.center.label == _doc.label:
				return cluster
		return self.NEW_CLUSTER

class AlmostPerfectCONFIRM(PerfectCONFIRM):
	'''
	It's perfect (1-p)% of the time.  p% of the time, it makes a random cluster assignment
	At the end, it puts each doc back in the correct cluster.
	Used for testing robustness of representation/similarity when bad choices are made
	'''
	
	def __init__(self, p=0.05, **kwargs):
		super(AlmostPerfectCONFIRM, self).__init__(**kwargs)
		self.p = p
	
	def _choose_cluster(self, _doc):
		if random.random() > p:
			return super(AlmostPerfectCONFIRM, self)._choose_cluster(_doc)
		return random.sample(self.clusters, 1)[0]

class AlmostPerfect2CONFIRM(AlmostPerfectCONFIRM):

	def post_process_clusters(self, **kwargs):
		for cluster in self.clusters:
			renegade_docs = list()
			for _doc in cluster.members:
				if _doc.label != cluster.center.label:
					renegade_docs.append(_doc)
			for _doc in renegade_docs:
				cluster.members.remove(_doc)
				for new_cluster in self.clusters:
					if new_cluster.center.label == _doc.label:
						new_cluster.members.append(_doc)
		

class RegionalCONFIRM(BaseCONFIRM):

	def doc_similarity(self, doc1, doc2):
		'''
		Basically looks at the region scores and weights them by how
			much feature mass is in that region.  Treats each feature
			type uniformly.
		'''
		region_scores_by_name = doc1.similarity_mats_weights_by_name(doc2)
		composite_regional_score = 0
		for sim_mat, weight_mat in region_scores_by_name.values():
			combined = utils.mult_mats([sim_mat, weight_mat])
			s = sum(map(sum, combined))  # add up all entries in mat
			composite_regional_score += s * 1.0 / len(region_scores_by_name)
		return composite_regional_score

class PerfectRegionalCONFIRM(RegionalCONFIRM, PerfectCONFIRM):
	pass

# inherits from regional confrim to get doc_similarity() for get_doc_sim_mat()
class RegionalWeightedCONFIRM(RegionalCONFIRM):
	'''
	Does automatic weighting of regions.  Feature weights are the sum of the feature
		sim scores of the documents belonging to that cluster
	'''

	def _uniform_mat(self, rows, cols):
		mat = [[1] * cols for x in xrange(rows)]
		return mat

	def _add_cluster(self, _doc, member=True):
		cluster = super(RegionalWeightedCONFIRM, self)._add_cluster(_doc, member)
		cluster.region_weights = {metric: self._uniform_mat(doc.ROWS, doc.COLS) for metric in _doc.similarity_function_names()}
		cluster.global_weights = {metric: 1 for metric in _doc.similarity_function_names()}

	def cluster_doc_similarity(self, cluster, _doc):
		sim_mats = cluster.center.similarity_mats_weights_by_name(_doc)
		sim_scores = cluster.center.similarities_by_name(_doc)
		composite_global_score = utils.wavg(sim_scores.values(), utils.norm_list(cluster.global_weights.values()))
		composite_regional_score = 0
		for name in sim_mats:
			mat, weights = sim_mats[name]
			combined = utils.mult_mats([mat, utils.norm_mat(cluster.region_weights[name])])
			s = sum(map(sum, combined))  # add up all entries in mat
			composite_regional_score += s * 1.0 / len(sim_mats)
		return (composite_global_score + composite_regional_score) / 2

	def _add_to_cluster(self, cluster, _doc):
		super(RegionalWeightedCONFIRM, self)._add_to_cluster(cluster, _doc)

		# add in scores to get weights
		sim_scores = cluster.center.similarities_by_name(_doc)
		sim_mats = cluster.center.similarity_mats_weights_by_name(_doc)
		for name in sim_scores:
			cluster.global_weights[name] += sim_scores[name]
			sim_mat = utils.mult_mats(sim_mats[name])
			weight_mat = cluster.region_weights[name]
			for r in xrange(len(weight_mat)):
				for c in xrange(len(weight_mat[r])):
					weight_mat[r][c] += sim_mat[r][c]

class PerfectRegionalWeightedCONFIRM(RegionalWeightedCONFIRM, PerfectCONFIRM):
	pass

class WavgNetCONFIRM(RegionalCONFIRM):
	'''
	Uses a linear classifier to learn feature weights and calculate similarity.
	Backprop with SGD updates.
	Handles "empty" inputs by not including their weights in the normalization.
	Each cluster's network only sees positive examples (no negative competition)
	'''

	def __init__(self, docs, lr, **kwargs):
		super(WavgNetCONFIRM, self).__init__(docs, **kwargs)
		self.lr = lr

	def _add_cluster(self, _doc, member=False):
		cluster = super(WavgNetCONFIRM, self)._add_cluster(_doc, member)
		weights = _doc.get_initial_vector_weights(_doc)
		cluster.network = network.WeightedAverageNetwork(len(weights), weights, self.lr)

	def cluster_doc_similarity(self, cluster, _doc):
		sim_vec = cluster.center.similarity_vector(_doc)
		return cluster.network.wavg(sim_vec)
		
	def _add_to_cluster(self, cluster, _doc):
		super(WavgNetCONFIRM, self)._add_to_cluster(cluster, _doc)
		sim_vec = cluster.center.similarity_vector(_doc)
		cluster.network.learn(sim_vec, 1)

	def get_cluster_sim_mat(self):
		mat = []
		for clust1 in self.clusters:
			row = []
			for clust2 in self.clusters:
				if clust1 == clust2:
					row.append(1.0)
				else:
					row.append(self.cluster_doc_similarity(clust1, clust2.center))
			mat.append(row)
		return mat


class CompetitiveWavgNetCONFIRM(WavgNetCONFIRM):
	'''
	Like WavgNetCONFIRM, but instead of only updating the network of the assinged cluster,
		the next closest cluster is decremented.
	'''
	def _add_to_cluster(self, cluster, _doc):
		super(WavgNetCONFIRM, self)._add_to_cluster(cluster, _doc)
		sim_vec = cluster.center.similarity_vector(_doc)
		cluster.network.learn(sim_vec, 1)

		# competitive stage
		similarities = self._cluster_sim_scores(_doc)
		idx = utils.argmax(similarities)
		del similarities[idx]
		if similarities:
			idx2 = utils.argmax(similarities)
			if idx2 <= idx:
				idx2 += 1
			sim_vec2 = self.clusters[idx2].center.similarity_vector(_doc)
			self.clusters[idx2].network.learn(sim_vec2, 0.2)
		

class PerfectCompetitiveWavgCONFIRM(CompetitiveWavgNetCONFIRM, PerfectCONFIRM):
	pass

class MSTInitCONFIRM(BaseCONFIRM):
	'''
	Initializes the set of clusters by taking the first $num_instances docs and forms
		a minimal spanning tree from their similarity matrix.  Then random edges are removed until
		the largest connected component is less than $num_init.  Then the nodes in that component
		are used as initial cluster centers
	Note that this isn't a very good approx because vertices in the largest cc that aren't joined
		by an edge in the MST can be very similar to each other.
	'''

	def __init__(self, docs, num_init=5, num_instances=20, **kwargs):
		super(MSTInitCONFIRM, self).__init__(docs, **kwargs)
		self.num_init = num_init
		self.num_instances = num_instances

	def _init_clusters(self):
		sub_docs = self.docs[:self.num_instances]
		sim_mat = utils.pairwise(sub_docs, 
			lambda x, y: max(self.doc_similarity(x, y), self.doc_similarity(y, x)))

		edges = utils.minimum_spanning_tree(sim_mat)
		ccs = utils.get_ccs(range(self.num_instances), edges) 
		biggest_cc = max(map(len, ccs))
		while biggest_cc > self.num_init:
			edge_to_remove = random.sample(edges, 1)[0]
			edges.remove(edge_to_remove)
			ccs = utils.get_ccs(range(self.num_instances), edges)
			biggest_cc = max(map(len, ccs))
		cc = ccs[utils.argmax(map(len, ccs))]

		for idx in cc:
			self._add_cluster(self.docs[idx], member=False)


class MaxCliqueInitCONFIRM(BaseCONFIRM):
	'''
	Initializes the set of clusters by finding a clique of size $num_clust that minimizes
		max of any weight between them
	'''
	
	def __init__(self, docs, num_clust=5, num_instances=20, **kwargs):
		super(MaxCliqueInitCONFIRM, self).__init__(docs, **kwargs)
		self.num_clust = num_clust
		self.num_instances = num_instances

	def _init_clusters(self):
		sub_docs = self.docs[:self.num_instances]
		sim_mat = utils.pairwise(sub_docs, 
			lambda x, y: max(self.doc_similarity(x, y), self.doc_similarity(y, x)))

		#print
		#print "Doc Sim Mat"
		#utils.print_mat(utils.apply_mat(sim_mat, lambda x: "%3.2f" % x))

		idxs = utils.find_best_clique(sim_mat, self.num_clust)

		#print 
		#print "Cluster Labels:"
		for idx in idxs:
			self._add_cluster(self.docs[idx], member=False)
			#print idx, self.docs[idx].label

class MaxClustersCONFIRM(BaseCONFIRM):
	'''
	Doesn't create a new cluster after maxK clusters has been reached
	'''
	
	def __init__(self, docs, maxK=25, **kwargs):
		super(MaxClustersCONFIRM, self).__init__(docs, **kwargs)
		self.maxK=25

	def _sufficiently_similar(self, _doc, cluster, sim_score, **kwargs):
		return sim_score > self.sim_thresh or len(self.clusters) > self.maxK
		


class InfoCONFIRM(BaseCONFIRM):
	
	def _after_iteration(self, _doc, new_cluster, **kwargs):
		print
		if new_cluster:
			cluster, similarity = self._most_similar_cluster(_doc)
			print "New Cluster Top Sim: %.2f\t%s" % (similarity, _doc.label)
		else:
			sim_scores = self._get_cached_sim_scores(_doc)
			cluster, similarity = self._most_similar_cluster(_doc)
			cluster.set_label()
			margin = max(map(lambda score: similarity - score if similarity != score else -1, sim_scores))
			print "%d\t%s\t%s\t%s\t%.2f\t%.2f\t%s" % (self.num_clustered, _doc.label, _doc.label == cluster.label, 
													cluster.label, 
													similarity, margin, " ".join(map(lambda x: "%.2f" % x, sim_scores)))
		self.display_weights()

	def display_weights(self):
		for x, cluster in enumerate(self.clusters):
			print "%s\t%s" % (x, " ".join(map(lambda w: "%.3f" % w, cluster.network.weights)))
		

class AdaptiveThresholdCONFIRM(MaxCliqueInitCONFIRM):
	'''
	Each cluster has a similarity threshold that is interpolated with a global threshold
		based on how many instances are in the cluster
	'''

	def __init__(self, docs, A=10, N=10, **kwargs):
		super(AdaptiveThresholdCONFIRM, self).__init__(docs, **kwargs)
		self.A = A
		self.N = N
		self.sim_sum = 0
		self.num_sims = 0

	def _get_cluster_thresh(self, cluster):
		num = max(1, len(cluster.members))
		l = 2 ** (- (num - 1) / self.A)
		return l * self.global_thresh + (1 - l) * cluster.local_thresh

	def _calc_local_thresh(self, cluster):
		mean = utils.avg(cluster.recent_sim_scores)
		#std_dev = utils.stddev(cluster.recent_sim_scores, mean)
		#cluster.local_thresh = mean - 0.1
		cluster.local_thresh = mean * 0.9

	def _init_clusters(self):
		super(AdaptiveThresholdCONFIRM, self)._init_clusters()
		sim_mat = self.get_cluster_sim_mat()
		self.global_thresh = max(map(max, sim_mat))

	def _add_cluster(self, _doc, member=True):
		cluster = super(AdaptiveThresholdCONFIRM, self)._add_cluster(_doc, member)
		cluster.local_thresh = 0
		cluster.recent_sim_scores = list()
	
	def _add_to_cluster(self, cluster, _doc):
		super(AdaptiveThresholdCONFIRM, self)._add_to_cluster(cluster, _doc)
		cluster.recent_sim_scores.append(self._cached_most_similar_val)
		if len(cluster.recent_sim_scores) > self.N:
			cluster.recent_sim_scores.pop(0)
		self._calc_local_thresh(cluster)
		self._set_global_thresh()

	def _set_global_thresh(self):
		self.sim_sum += self._cached_most_similar_val
		self.num_sims += 1
		self.global_thresh = 0.9 * (self.sim_sum / float(self.num_sims))

	def _sufficiently_similar(self, _doc, cluster, sim_score, **kwargs):
		return sim_score > self._get_cluster_thresh(cluster)

##### Doesn't work.  The std lib for multiprocessing uses cPickle for tranfering
##### functions to worker threads.  cPickle cannot handle closures or lambdas...
class ParallelCONFIRM(BaseCONFIRM):
	
	def __init__(self, docs, processes=4, **kwargs):
		super(ParallelCONFIRM, self).__init__(docs, **kwargs)
		self.num_processes = processes
		self.pool = multiprocessing.Pool(processes=processes)

	def _calc_sim_scores(self, _doc):
		this = self
		print len(self.clusters)
		def calc_single_val(cluster):
			return this.cluster_doc_similarity(cluster, _doc)
		tmp =  self.pool.map(calc_single_val, self.clusters)#, len(self.clusters) / self.num_processes + 1)
		print tmp
		return tmp

	def get_cluster_sim_mat(self):
		mat = [ [0] * (len(self.clusters)) for _ in xrange(len(self.clusters))]
		confirm = self
		nclusters = len(self.clusters)
		def calc_single_val(val): 
			i = val / nclusters
			j = val % nclusters
			if i == j:
				mat[i][j] = 1.0
			else:
				mat[i][j] = confirm.cluster_similarity(confirm.clusters[i], confirm.clusters[j])
		self.pool.map(calc_single_val, xrange(nclusters ** 2), 100)
		return mat

	# may be expensive
	def get_doc_cluster_sim_mat(self):
		mat = [ [0] * (len(self.clusters)) for _ in xrange(len(self.docs))]
		confirm = self
		nclusters = len(self.clusters)
		def calc_single_val(val): 
			i = val / nclusters
			j = val % nclusters
			if i == j:
				mat[i][j] = 1.0
			else:
				mat[i][j] = confirm.cluster_doc_similarity(confirm.clusters[j], confirm.docs[i])
		self.pool.map(calc_single_value, xrange(nclusters * len(self.docs)), 100)
		return mat
		
		
class TestCONFIRM(WavgNetCONFIRM, MaxCliqueInitCONFIRM, RedistributePruningCONFIRM, TwoPassCONFIRM, InfoCONFIRM):
	pass

class BestCONFIRM(WavgNetCONFIRM, MaxCliqueInitCONFIRM, RedistributePruningCONFIRM, 
				MaxClustersCONFIRM, InfoCONFIRM, TwoPassCONFIRM):
	pass

class BestPerfectCONFIRM(PerfectCONFIRM):
	pass


