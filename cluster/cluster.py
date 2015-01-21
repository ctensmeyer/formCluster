
import network
import utils
import doc

import multiprocessing
import collections
import itertools
import random
from constants import *

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
		#print "Base _add_to_cluster()"
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
		return utils.harmonic_mean_list(doc1.global_sim(doc2))

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

	def __init__(self, docs, min_size, **kwargs):
		super(PruningCONFIRM, self).__init__(docs, **kwargs)
		self.min_size = min_size
	
	def post_process_clusters(self, **kwargs):
		''' 
		Prune all clusters of size < minsize
		:return: list of docs that were members of the pruned clusters
		'''
		super(PruningCONFIRM, self).post_process_clusters(**kwargs)
		odd_docs = list()
		clusters_to_remove = list()
		for cluster in self.clusters:
			if len(cluster.members) < self.min_size:
				odd_docs += cluster.members
				clusters_to_remove.append(cluster)
		for cluster in clusters_to_remove:
			self.clusters.remove(cluster)
		return odd_docs

class IsolatePruningCONFIRM(PruningCONFIRM):
	
	def post_process_clusters(self, **kwargs):
		''' 
		Take all docs in clusters of size < minsize and stick them in a single isolated cluster
		'''
		odd_docs = super(IsolatePruningCONFIRM, self).post_process_clusters(**kwargs)
		if odd_docs:
			# make a single cluster of the oddballs
			odd_cluster = self._add_cluster(odd_docs[0])
			for _doc in odd_docs[1:]:
				odd_cluster.members.append(_doc)

class RedistributePruningCONFIRM(PruningCONFIRM):
	
	def post_process_clusters(self, **kwargs):
		''' 
		Take all docs in clusters of size < minsize and assign them to the most similar cluster
			of size >= minsize
		'''
		odd_docs = super(RedistributePruningCONFIRM, self).post_process_clusters(**kwargs)
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
		

class RegionCONFIRM(BaseCONFIRM):

	def doc_similarity(self, doc1, doc2):
		'''
		Basically looks at the region scores and weights them by how
			much feature mass is in that region.  Treats each feature
			type uniformly.
		'''
		region_sim_weights = doc1.region_sim_weights(doc2)
		composite_region_score = 0
		for sim_mat, weight_mat in region_sim_weights:
			combined = utils.mult_mats([sim_mat, weight_mat])
			s = sum(map(sum, combined))  # add up all entries in mat
			composite_region_score += s * 1.0 / len(region_sim_weights)
		return composite_region_score

class PerfectRegionCONFIRM(RegionCONFIRM, PerfectCONFIRM):
	pass

# inherits from region confrim to get doc_similarity() for get_doc_sim_mat()
class RegionWeightedCONFIRM(RegionCONFIRM):
	'''
	Does automatic weighting of regions.  Feature weights are the sum of the feature
		sim scores of the documents belonging to that cluster
	'''

	def _uniform_mat(self, rows, cols):
		mat = [[1] * cols for x in xrange(rows)]
		return mat

	def _add_cluster(self, _doc, member=True):
		cluster = super(RegionWeightedCONFIRM, self)._add_cluster(_doc, member)
		l = len(_doc.get_feature_set_names())
		cluster.region_weights = [self._uniform_mat(REGION_ROWS, REGION_COLS) for metric in xrange(l)]
		cluster.global_weights = [1] * l
		return cluster

	def cluster_doc_similarity(self, cluster, _doc):
		region_sim_weights = cluster.center.region_sim_weights(_doc)
		global_sims = cluster.center.global_sim(_doc)
		composite_global_score = utils.wavg(global_sims, utils.norm_list(cluster.global_weights))
		composite_region_score = 0
		for x in xrange(len(region_sim_weights)):
			mat, weights = region_sim_weights[x]
			combined = utils.mult_mats([mat, utils.norm_mat(cluster.region_weights[x])])
			s = sum(map(sum, combined))  # add up all entries in mat
			composite_region_score += s * 1.0 / len(region_sim_weights)
		return (composite_global_score + composite_region_score) / 2

	def _add_to_cluster(self, cluster, _doc):
		super(RegionWeightedCONFIRM, self)._add_to_cluster(cluster, _doc)

		# add in scores to get weights
		global_sims = cluster.center.global_sim(_doc)
		region_sim_weights = cluster.center.region_sim_weights(_doc)
		for x in xrange(len(global_sims)):
			cluster.global_weights[x] += global_sims[x]
			sim_mat = utils.mult_mats(region_sim_weights[x])
			weight_mat = cluster.region_weights[x]
			for r in xrange(len(weight_mat)):
				for c in xrange(len(weight_mat[r])):
					weight_mat[r][c] += sim_mat[r][c]

class PerfectRegionWeightedCONFIRM(RegionWeightedCONFIRM, PerfectCONFIRM):
	pass

class WavgNetCONFIRM(RegionCONFIRM):
	'''
	Uses a linear classifier to learn feature weights and calculate similarity.
	Backprop with SGD updates.
	Handles "empty" inputs by not including their weights in the normalization.
	Each cluster's network only sees positive examples (no negative competition)
	'''

	def __init__(self, docs, lr, **kwargs):
		super(WavgNetCONFIRM, self).__init__(docs, **kwargs)
		self.lr = lr

	def _add_cluster(self, _doc, member=True):
		cluster = super(WavgNetCONFIRM, self)._add_cluster(_doc, member)
		weights = _doc.global_region_weights()
		cluster.network = network.WeightedAverageNetwork(len(weights), weights, self.lr)
		return cluster

	def cluster_doc_similarity(self, cluster, _doc):
		sim_vec = cluster.center.global_region_sim(_doc)
		return cluster.network.wavg(sim_vec)
		
	def _add_to_cluster(self, cluster, _doc):
		super(WavgNetCONFIRM, self)._add_to_cluster(cluster, _doc)
		#print "WavgNet _add_to_cluster()"
		sim_vec = cluster.center.global_region_sim(_doc)
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

class PerfectWavgNetCONFIRM(WavgNetCONFIRM, PerfectCONFIRM):
	pass

class CompetitiveWavgNetCONFIRM(WavgNetCONFIRM):
	'''
	Like WavgNetCONFIRM, but instead of only updating the network of the assinged cluster,
		the next closest cluster is decremented.
	'''
	def _add_to_cluster(self, cluster, _doc):
		super(WavgNetCONFIRM, self)._add_to_cluster(cluster, _doc)

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
		

class PushAwayCONFIRM(BaseCONFIRM):
	
	def _add_to_cluster(self, cluster, _doc):
		super(PushAwayCONFIRM, self)._add_to_cluster(cluster, _doc)

		sim_score = self._cached_most_similar_val
		margins = map(lambda x: sim_score - x if sim_score != x else 0, self._get_cached_sim_scores(_doc))
		most_similar_cluster = self.clusters[utils.argmax(margins)]
		cluster.center.push_away(most_similar_cluster.center)


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
		super(MaxCliqueInitCONFIRM, self)._init_clusters()
		sub_docs = self.docs[:self.num_instances]
		sim_mat = utils.pairwise(sub_docs, 
			lambda x, y: max(self.doc_similarity(x, y), self.doc_similarity(y, x)))

		print
		print "Doc Sim Mat"
		utils.print_mat(utils.apply_mat(sim_mat, lambda x: "%3.2f" % x))

		idxs = utils.find_best_clique(sim_mat, self.num_clust)

		print 
		print "Cluster Labels:"
		for idx in idxs:
			self._add_cluster(self.docs[idx], member=False)
			print idx, self.docs[idx].label

class SupervisedInitCONFIRM(BaseCONFIRM):
	'''
	Uses the labels to initialize clusters
	'''

	def __init__(self, docs, instances_per_cluster=3, **kwargs):
		super(SupervisedInitCONFIRM, self).__init__(docs, **kwargs)
		self.instances_per_cluster = instances_per_cluster

	def _init_clusters(self):
		super(SupervisedInitCONFIRM, self)._init_clusters()
		counter = collections.defaultdict(int)
		self.used_docs = list()
		label_to_cluster = dict()
		for _doc in self.docs:
			_doc._load_check()
			label = _doc.label
			if counter[label] == 0:
				cluster = self._add_cluster(_doc, member=True)
				counter[label] += 1
				self.used_docs.append(_doc)
				label_to_cluster[label] = cluster
			elif counter[label] < self.instances_per_cluster:
				self._add_to_cluster(label_to_cluster[label], _doc)
				counter[label] += 1
				self.used_docs.append(_doc)
		for _doc in self.used_docs:
			self.docs.remove(_doc)
		self.num_clustered = len(self.used_docs)

	def post_process_clusters(self):
		self.docs = self.used_docs + self.docs
		super(SupervisedInitCONFIRM, self).post_process_clusters()
		

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
		print map(lambda x: len(x.members), self.clusters)
		if hasattr(self, '_get_global_thresh'):
			print "Global Thresh:", self._get_global_thresh()
		if new_cluster:
			cluster, similarity = self._most_similar_cluster(_doc)
			print "%d New Cluster Top Sim: %.2f\t%s\t%s" % (self.num_clustered, similarity, _doc.label, _doc._id)
		else:
			sim_scores = self._get_cached_sim_scores(_doc)
			cluster, similarity = self._most_similar_cluster(_doc)
			cluster.set_label()
			margin = min(map(lambda score: similarity - score if similarity != score else 1, sim_scores))
			print "%d\t%s\t%s\t%s\t%d\t%.2f\t%.2f\t%s" % (self.num_clustered, _doc.label, _doc.label == cluster.label, 
													cluster.label, self.clusters.index(cluster),
													similarity, margin, " ".join(map(lambda x: "%.2f" % x, sim_scores)))
		self.display_weights()

	def display_weights(self):
		for x, cluster in enumerate(self.clusters):
			if hasattr(cluster, 'network'):
				print "%s\t%s" % (x, " ".join(map(lambda w: "%.3f" % w, cluster.network.weights)))
			if hasattr(cluster, 'local_thresh'):
				print "%.2f\t%.2f" % (cluster.local_thresh, self._get_interpolated_thresh(cluster))
		

class InterpolatedThresholdCONFIRM(BaseCONFIRM):
	'''
	This CONFIRM uses an Adaptive threshold that is interpolated between a global
		threshold and a threshold local to each cluster.  The interpolation coefficient
		of the global threshold has a half life of $A.
	The particular global and local thresholds are defined in subclasses.
	'''

	def __init__(self, docs, A=10, **kwargs):
		super(InterpolatedThresholdCONFIRM, self).__init__(docs, **kwargs)
		self.A = A

	# performs the interpolation
	def _get_interpolated_thresh(self, cluster):
		num = max(1, len(cluster.members))
		l = 2 ** (- (num - 1) / float(self.A))
		return l * self._get_global_thresh() + (1 - l) * self._get_cluster_thresh(cluster)

	def _init_clusters(self):
		super(InterpolatedThresholdCONFIRM, self)._init_clusters()
		self._init_global_thresh()

	def _add_cluster(self, _doc, member=True):
		cluster = super(InterpolatedThresholdCONFIRM, self)._add_cluster(_doc, member)
		self._init_cluster_thresh(cluster)
		return cluster

	# this is when updates to thresholds can occur
	def _add_to_cluster(self, cluster, _doc):
		super(InterpolatedThresholdCONFIRM, self)._add_to_cluster(cluster, _doc)
		self._update_cluster_thresh(cluster, _doc)
		self._update_global_thresh(cluster, _doc)

	def _sufficiently_similar(self, _doc, cluster, sim_score, **kwargs):
		return sim_score > self._get_interpolated_thresh(cluster)
	
	# The following methods must be overwritten
	def _get_global_thresh(self):
		pass

	def _get_cluster_thresh(self, cluster):
		pass

	def _init_global_thresh(self):
		pass

	def _init_cluster_thresh(self, cluster):
		pass
	
	def _update_cluster_thresh(self, cluster, _doc):
		pass

	def _update_global_thresh(self, cluster, _doc):
		pass

class OneNNInitGlobalThresholdCONFIRM(InterpolatedThresholdCONFIRM):

	def __init__(self, docs, initial_thresh_weight=1, **kwargs):
		super(OneNNInitGlobalThresholdCONFIRM, self).__init__(docs, **kwargs)
		self.initial_thresh_weight = initial_thresh_weight
	
	def _init_global_thresh(self):
		sub_docs = self.docs[:20]
		sim_mat = utils.pairwise(sub_docs, 
			lambda x, y: max(self.doc_similarity(x, y), self.doc_similarity(y, x)))
		for x in xrange(len(sim_mat)):
			del sim_mat[x][x]
		self.global_thresh = .7 #utils.avg(map(max, sim_mat))
		print
		print "INITIAL GLOBAL THRESH: ", self.global_thresh
		tmp = utils.flatten(sim_mat)
		tmp.sort()
		print map(lambda x: "%.2f" % x, tmp)
		print map(lambda x: "%.2f" % x, map(max, sim_mat))
		print
		self.sim_sum = self.global_thresh * self.initial_thresh_weight
		self.num_counted = self.initial_thresh_weight

class GlobalMarginThresholdCONFIRM(OneNNInitGlobalThresholdCONFIRM):

	def _init_global_thresh(self):
		super(GlobalMarginThresholdCONFIRM, self)._init_global_thresh()
		self.margin_sum = 0

	def _get_global_thresh(self):
		return self.global_thresh
	
	def _update_global_thresh(self, cluster, _doc):
		sim_score = self._cached_most_similar_val
		margin = max(map(lambda x: sim_score - x if sim_score != x else 0, self._get_cached_sim_scores(_doc)))
		self.sim_sum += sim_score
		self.margin_sum += margin
		self.num_counted += 1
		self.global_thresh = (self.sim_sum - self.margin_sum) / self.num_counted


class LocalMarginThresholdCONFIRM(InterpolatedThresholdCONFIRM):

	def _init_cluster_thresh(self, cluster):
		cluster.local_thresh = 0
		cluster.sim_sum = 0
		cluster.margin_sum = 0

	def _get_cluster_thresh(self, cluster):
		return cluster.local_thresh

	def _update_cluster_thresh(self, cluster, _doc):
		sim_score = self._cached_most_similar_val
		margin = max(map(lambda x: sim_score - x if sim_score != x else 0, self._get_cached_sim_scores(_doc)))
		cluster.sim_sum += sim_score
		cluster.margin_sum += margin
		cluster.local_thresh = (cluster.sim_sum - cluster.margin_sum) / len(cluster.members) 

class MarginThresholdCONFIRM(LocalMarginThresholdCONFIRM, GlobalMarginThresholdCONFIRM):
	pass

class GlobalSimMultThresholdCONFIRM(OneNNInitGlobalThresholdCONFIRM):

	def __init__(self, docs, global_thresh_mult=0.9, **kwargs):
		super(GlobalSimMultThresholdCONFIRM, self).__init__(docs, **kwargs)
		self.global_thresh_mult = global_thresh_mult
	
	def _get_global_thresh(self):
		return self.global_thresh
	
	def _update_global_thresh(self, cluster, _doc):
		if hasattr(self, '_cached_most_similar_val'):
			sim_score = self._cached_most_similar_val
		else:
			sim_score = self.cluster_doc_similarity(cluster, _doc)
			self._cached_most_similar_val = sim_score
		self.sim_sum += sim_score
		self.num_counted += 1
		self.global_thresh = self.global_thresh_mult * self.sim_sum / self.num_counted

class LocalSimMultThresholdCONFIRM(InterpolatedThresholdCONFIRM):
	
	def __init__(self, docs, local_thresh_mult=0.9, **kwargs):
		super(LocalSimMultThresholdCONFIRM, self).__init__(docs, **kwargs)
		self.local_thresh_mult = local_thresh_mult

	def _init_cluster_thresh(self, cluster):
		cluster.local_thresh = 0
		cluster.sim_sum = 0

	def _get_cluster_thresh(self, cluster):
		return cluster.local_thresh

	def _update_cluster_thresh(self, cluster, _doc):
		if hasattr(self, '_cached_most_similar_val'):
			sim_score = self._cached_most_similar_val
		else:
			sim_score = self.cluster_doc_similarity(cluster, _doc)
			self._cached_most_similar_val = sim_score
		cluster.sim_sum += sim_score
		cluster.local_thresh = self.local_thresh_mult * cluster.sim_sum / len(cluster.members) 

class SimMultThresholdCONFIRM(GlobalSimMultThresholdCONFIRM, LocalSimMultThresholdCONFIRM):
	pass


class FastCONFIRM(BaseCONFIRM):
	'''
	This version of CONFIRM doesn't make outliers first class clusters immediately.  It pools them
		separately until it can find $min_membership.  The majority of cases will just consider
		the set of big clusters, which should remain small.
	'''
	
	def __init__(self, docs, min_membership=5, **kwargs):
		super(FastCONFIRM, self).__init__(docs, **kwargs)
		self.min_membership = min_membership
		self.potential_clusters = list()

	def handle_reject(self, _doc):
		# treat the potential clusters like the real clusters and do a clustering step
		tmp = self.clusters
		self.clusters = self.potential_clusters
		self._cached_doc = None

		new_cluster = False
		if not self.clusters:
			self._add_cluster(_doc)
		else:
			cluster = self._choose_cluster(_doc)
			if cluster == self.NEW_CLUSTER:
				new_cluster = self._add_cluster(_doc)
			else:
				self._add_to_cluster(cluster, _doc)
				if len(cluster.members) >= self.min_membership:
					self.potential_clusters.remove(cluster)
					tmp.append(cluster)
					new_cluster = True
		# reset the aliases
		self.clusters = tmp
		return new_cluster
		

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
					new_cluster = self.handle_reject(_doc)
				else:
					self._add_to_cluster(cluster, _doc)
			self.num_clustered += 1
			self._after_iteration(_doc, new_cluster=new_cluster)
		self.post_process_clusters()

	def _after_iteration(self, _doc, new_cluster):
		print
		if new_cluster:
			print "New Cluster:"
		print map(lambda x: len(x.members), self.clusters)
		print map(lambda x: len(x.members), self.potential_clusters)

	def post_process_clusters(self):
		self.clusters = self.clusters + self.potential_clusters
		super(FastCONFIRM, self).post_process_clusters()

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
		
		
class TestCONFIRM(MaxCliqueInitCONFIRM, RedistributePruningCONFIRM, TwoPassCONFIRM, InfoCONFIRM):
	pass

class BaseTestCONFIRM(TestCONFIRM):
	pass

class RegionTestCONFIRM(RegionCONFIRM, TestCONFIRM):
	pass

class RegionWeightedTestCONFIRM(RegionWeightedCONFIRM, TestCONFIRM):
	pass

class WavgNetTestCONFIRM(WavgNetCONFIRM, TestCONFIRM):
	pass

class BestCONFIRM(PushAwayCONFIRM, TestCONFIRM):
	pass

class BestPerfectCONFIRM(PerfectCONFIRM):
	pass


