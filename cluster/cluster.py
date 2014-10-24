
import network
import utils
import doc

import collections
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
		self.label = c.most_common(1)[0][0]


class BaseCONFIRM(object):
	
	NEW_CLUSTER = -1
	def __init__(self, docs, sim_thresh=None, **kwargs):
		self.docs = docs
		self.clusters = list()
		self.num_clustered = 0
		self.sim_thresh = sim_thresh

	def _before_iteration(self, **kwargs):
		pass

	def _after_iteration(self, **kwargs):
		if self.num_clustered % 10 == 0:
			print "%d documents processed" % self.num_clustered

	def _add_cluster(self, _doc, member=True):
		prototype = _doc.copy(len(self.clusters))
		prototype.label = None
		members = list()
		if member:
			member.append(_doc)
		cluster = Cluster(members, prototype)
		self.clusters.append(cluster)
		return cluster

	def _add_to_cluster(self, cluster, _doc):
		cluster.center.aggregate(_doc)
		cluster.members.append(_doc)

	def _sufficiently_similar(self, _doc, cluster, sim_score, **kwargs):
		return sim_score > self.sim_thresh

	def _cluster_sim_scores(self, _doc):
		return map(lambda cluster: self.cluster_doc_similarity(cluster, _doc), self.clusters)

	def _most_similar_cluster(self, _doc):
		similarities = self._cluster_sim_scores(_doc)
		idx = utils.argmax(similarities)
		cluster = self.clusters[idx]
		similarity = similarities[idx]
		return cluster, similarity

	def _choose_cluster(self, _doc):
		cluster, similarity = self._most_similar_cluster(_doc)
		if self._sufficiently_similar(_doc, cluster, similarity):
			return cluster
		else:
			return self.NEW_CLUSTER

	def _init_clusters():
		pass

	def cluster(self):
		self._init_clusters()
		for x, _doc in enumerate(self.docs):
			self._before_iteration()
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
			self._after_iteration(new_cluster=new_cluster)
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

	def _after_iteration(self, **kwargs):
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

	def _add_cluster(self, _doc):
		cluster = super(RegionalWeightedCONFIRM, self)._add_cluster(_doc)
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

	def _add_cluster(self, _doc):
		cluster = super(WavgNetCONFIRM, self)._add_cluster(_doc)
		weights = _doc.get_initial_vector_weights(_doc)
		cluster.network = network.WeightedAverageNetwork(len(weights), weights, self.lr)

	def cluster_doc_similarity(self, cluster, _doc):
		sim_vec = cluster.center.similarity_vector(_doc)
		return cluster.network.wavg(sim_vec)
		
	def _add_to_cluster(self, cluster, _doc):
		super(WavgNetCONFIRM, self)._add_to_cluster(cluster, _doc)
		sim_vec = cluster.center.similarity_vector(_doc)
		cluster.network.learn(sim_vec, 1)
	
class PerfectWavgCONFIRM(WavgNetCONFIRM, PerfectCONFIRM):
	pass

class MSTInitCONFIRM(BaseCONFIRM):
	'''
	Initializes the set of clusters by taking the first $num_instances docs and forms
		a maximal spanning tree from their similarity matrix.  Then random edges are removed to
		form $num_init connected components (ccs).  Then one doc is sampled from each cc to be
		an initial cluster.
	'''

	def __init__(self, docs, num_init=5, num_instances=20, **kwargs):
		super(MSTInitCONFIRM, self).__init__(docs, **kwargs)
		self.num_init = num_init
		self.num_instances = num_instances

	def _init_clusters(self):
		sub_docs = docs[:self.num_instances]
		sim_mat = utils.pairwise(sub_docs, lambda x, y: max(self.doc_similarity(x, y), self.doc_similarity(y, x)))
		edges = utils.maximal_spanning_tree(sim_mat)
		to_remove = random.sample(edges, self.num_init - 1)
		for edge in to_remove:
			edges.remove(edge)
		ccs = utils.get_ccs(edges)
		indices = map(lambda cc: random.sample(cc, 1)[0], ccs)
		for idx in indices:
			self._add_cluster(self.docs[idx], member=False)
		
