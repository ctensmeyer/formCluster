
import metric
import cluster
import network
import numpy as np
import scipy.spatial.distance
import sklearn.ensemble
import sklearn.cluster
import collections
import utils
import random
import multiprocessing
from constants import *


def compute_random_matrix(data_matrix):
	print "Constructing Random Training Set"

	rand_shape = (int(data_matrix.shape[0] * SIZE_OF_RANDOM_DATA), data_matrix.shape[1])
	rand_mat = np.zeros(rand_shape)
	for col in xrange(rand_mat.shape[1]):
		vals = data_matrix[:,col]
		for row in xrange(rand_mat.shape[0]):
			rand_mat[row, col] = np.random.choice(vals)

	print "Done\n"
	return rand_mat

def train_random_forest(real_data, fake_data):
	print "Training Random Forest"

	print "Num features:", real_data.shape[1]
	num_tree_features = FUNCTION_NUM_FEATURES(real_data.shape[1])
	print "Computed Tree features:", num_tree_features
	rf = sklearn.ensemble.RandomForestClassifier(n_estimators=NUM_TREES, max_features=num_tree_features,
												bootstrap=False, n_jobs=RF_THREADS)
	combined_data = np.concatenate( (real_data, fake_data) )
	labels = np.concatenate( (np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])) )
	rf.fit(combined_data, labels)

	print "Done\n"
	return rf

def compute_sim_mat(data_matrix, random_forest):
	leaf_nodes = random_forest.apply(data_matrix)
	sim_mat = scipy.spatial.distance.pdist(leaf_nodes, "hamming")
	sim_mat = scipy.spatial.distance.squareform(sim_mat)
	sim_mat = 1 - sim_mat

	return sim_mat

def spectral_cluster(affinity_matrix, num_clusters):
	print "Performing Spectral Clustering"

	sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters , affinity="precomputed",
											assign_labels="discretize")
	assignments = sc.fit_predict(affinity_matrix)

	print "Done\n"
	return assignments

def spectral_cluster_cv(affinity_matrix, cluster_range):
	best_silhouette = -1
	best_assignments = list()
	dist_matrix = 1 - affinity_matrix

	print 
	print "Cross Validated Spectral Clustering"
	print
	for num_clusters in xrange(cluster_range[0], cluster_range[1] + 1):
		sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters , affinity="precomputed",
												assign_labels="discretize")
		assignments = sc.fit_predict(affinity_matrix)
		silhouette = sklearn.metrics.silhouette_score(dist_matrix, assignments, metric='precomputed')
		print num_clusters, "%.3f" % silhouette
		if silhouette > best_silhouette:
			best_silhouette = silhouette
			best_assignments = assignments
	
	return best_assignments

def spectral_cluster_all(affinity_matrix, cluster_range):
	all_assignments = dict()
	dist_matrix = 1 - affinity_matrix

	print 
	print "All Assignemnts Spectral Clustering"
	print
	for num_clusters in xrange(cluster_range[0], cluster_range[1] + 1):
		sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters , affinity="precomputed",
												assign_labels="discretize")
		assignments = sc.fit_predict(affinity_matrix)
		silhouette = sklearn.metrics.silhouette_score(dist_matrix, assignments, metric='precomputed')
		all_assignments[num_clusters] = (assignments, silhouette)
	
	return all_assignments

def form_clusters(instances, assignments):
	cluster_map = dict()
	class Mock:
		pass
	m = Mock()
	m.label = None
	for x in xrange(assignments.max() + 1):
		cluster_map[x] = cluster.Cluster(list(), m, x)
	for instance, assignment in zip(instances, assignments):
		cluster_map[assignment].members.append(instance)
	clusters = cluster_map.values()
	clusters = filter(lambda c: len(c.members), clusters)
	map(lambda cluster: cluster.set_label(), clusters)
	return clusters



def kumar_cluster(data_matrix, instances, num_clusters):
	'''
	data_matrix is a numpy matrix with one row for each instance's features
	instances are arbitrary objects that are clustered
	'''
	random_matrix = compute_random_matrix(data_matrix)
	rf = train_random_forest(data_matrix, random_matrix)
	sim_matrix = compute_sim_mat(data_matrix, rf)
	assignments = spectral_cluster(sim_matrix, num_clusters)
	clusters = form_clusters(instances, assignments)
	return clusters

def kumar_cluster_cv(data_matrix, instances, cluster_range):
	'''
	data_matrix is a numpy matrix with one row for each instance's features
	instances are arbitrary objects that are clustered
	'''
	random_matrix = compute_random_matrix(data_matrix)
	rf = train_random_forest(data_matrix, random_matrix)
	sim_matrix = compute_sim_mat(data_matrix, rf)
	assignments = spectral_cluster_cv(sim_matrix, cluster_range)
	clusters = form_clusters(instances, assignments)
	return clusters

	
class KumarCONFIRM(cluster.BaseCONFIRM):
	
	def __init__(self, docs, iterations=2, num_initial_seeds=10, num_seeds=10, cluster_range=(2,10), **kwargs):
		super(KumarCONFIRM, self).__init__(docs, **kwargs)
		self.num_initial_seeds = num_initial_seeds
		self.iterations = iterations
		self.num_seeds = num_seeds
		self.cluster_range = cluster_range

	def print_analysis(self):
		analyzer = metric.KnownClusterAnalyzer(self)
		analyzer.print_general_info()
		analyzer.print_histogram_info()
		analyzer.print_label_conf_mat()
		analyzer.print_label_cluster_mat()
		analyzer.print_label_info()
		analyzer.print_metric_info()

	def kumar_cluster_cv_cheat(self, data_matrix, instances, cluster_range):
		'''
		data_matrix is a numpy matrix with one row for each instance's features
		instances are arbitrary objects that are clustered
		'''
		random_matrix = compute_random_matrix(data_matrix)
		rf = train_random_forest(data_matrix, random_matrix)
		sim_matrix = compute_sim_mat(data_matrix, rf)
		all_assignments = spectral_cluster_all(sim_matrix, cluster_range)
		best_acc = 0
		best_assignments = 0
		for num_clusters in all_assignments:
			assignments, silhouette = all_assignments[num_clusters]
			self.clusters = form_clusters(instances, assignments)
			analyzer = metric.KnownClusterAnalyzer(self)
			accuracy = analyzer.accuracy()
			v_measure = analyzer.v_measure()
			print "%d: %2.1f%% %.3f %.3f" % (num_clusters, accuracy * 100, v_measure, silhouette)
			if accuracy > best_acc:
				best_acc = accuracy
				best_assignments = assignments
		clusters = form_clusters(instances, best_assignments)
		return clusters

	def cluster(self):
		seeds = self._choose_initial_seeds()
		for iteration in xrange(self.iterations):
			print 
			print "*" * 30
			print "ITERATION", iteration
			print "*" * 30
			print
			data_matrix = self._compute_features(seeds)
			if iteration < (self.iterations - 1):
				#self.clusters = self.kumar_cluster_cv_cheat(data_matrix, self.docs, self.cluster_range)
				self.clusters = kumar_cluster(data_matrix, self.docs, self.num_seeds)
				seeds = self._form_seeds()
			else:
				#self.clusters = self.kumar_cluster_cv_cheat(data_matrix, self.docs, self.cluster_range)
				self.clusters = kumar_cluster(data_matrix, self.docs, self.num_seeds)
			self.print_analysis()
				

	def _choose_initial_seeds(self):
		return self.docs[0:self.num_initial_seeds]

	def _form_seeds(self):
		seeds = list()
		for cluster in self.clusters:
			if not cluster.members:
				continue
			seed = cluster.members[0].copy()
			for _doc in cluster.members[1:]:
				seed.aggregate(_doc)
			seed.final_prune()
			seeds.append(seed)
		return seeds

	def _compute_features(self, seeds):
		num_features = 0
		_doc = self.docs[0]
		vectors = list()
		for seed in seeds:
			vectors.append(seed.match_vector(_doc))
		num_features = sum(map(len, vectors))
		feature_mat = np.zeros( (len(self.docs), num_features) )
		for x, _doc in enumerate(self.docs):
			offset = 0
			for seed in seeds:
				vector = seed.match_vector(_doc)
				feature_mat[x,offset:offset + len(vector)] = vector
				offset += len(vector)
		print feature_mat
		return feature_mat


class BatchMaxCliqueKumarCONFIRM(KumarCONFIRM):

	def __init__(self, docs, seeds_per_batch=2, batch_size=20, **kwargs):
		super(KumarCONFIRM, self).__init__(docs, **kwargs)
		self.seeds_per_batch = seeds_per_batch
		self.batch_size = batch_size

	def _choose_initial_seeds(self):
		print "Choosing seeds"
		seeds = list()
		batch_num = 0
		while len(seeds) < self.num_initial_seeds:
			batch = self.docs[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
			sim_mat = utils.pairwise(batch, 
				lambda x, y: max(self.doc_similarity(x, y), self.doc_similarity(y, x)))

			print
			print "Doc Sim Mat"
			utils.print_mat(utils.apply_mat(sim_mat, lambda x: "%3.2f" % x))
			idxs = utils.find_best_clique(sim_mat, self.seeds_per_batch)
			for idx in idxs:
				_doc = batch[idx]
				print "Adding %d %s" % (idx, _doc.label)
				seeds.append(_doc)
			batch_num += 1
		print "Done"
		return seeds
		

class SemiSupervisedKumarCONFIRM(KumarCONFIRM):
	
	def __init__(self, docs, num_per_seed=1, **kwargs):
		super(SemiSupervisedKumarCONFIRM, self).__init__(docs, **kwargs)
		self.num_per_seed = num_per_seed

	def _choose_initial_seeds(self):
		counter = collections.defaultdict(int)
		label_to_seed = dict()
		for _doc in self.docs:
			_doc._load_check()
			label = _doc.label
			if counter[label] == 0:
				seed = _doc.copy(label)
				counter[label] += 1
				label_to_seed[label] = seed
			elif counter[label] < self.num_per_seed:
				label_to_seed[label].aggregate(_doc)
				counter[label] += 1
		seeds = label_to_seed.values() 
		for seed in seeds:
			seed.final_prune()
		return seeds
		
class RandomSeedsKumarCONFIRM(KumarCONFIRM):
	
	def _choose_initial_seeds(self):
		return random.sample(self.docs, min(len(self.docs, self.num_initial_seeds)))


#def compute(args):
#	docs = args[0]
#	seeds = args[1]
#	start = args[2]
#	finish = args[3]
#	feature_mat = args[4]
#
#	# make copy of seeds so matching isn't thrown off
#	seeds = map(lambda seed: seed.copy(seed._id), seeds)
#	for x in xrange(start, finish):
#		print "Doc", x
#		offset = 0
#		_doc = docs[x]
#		_doc._load_check()
#		for seed in seeds:
#			vector = seed.match_vector(_doc)
#			feature_mat[x,offset:offset + len(vector)] = vector
#			offset += len(vector)
#	
#
#class ParallelKumarCONFIRM(KumarCONFIRM):
#	
#	def __init__(self, docs, processes=2, **kwargs):
#		super(ParallelKumarCONFIRM, self).__init__(docs, **kwargs)
#		self.processes = processes
#
#	def _compute_features(self, seeds):
#		num_features = 0
#		_doc = self.docs[0]
#		vectors = list()
#		for seed in seeds:
#			vectors.append(seed.match_vector(_doc))
#		num_features = sum(map(len, vectors))
#		pool = multiprocessing.Pool(self.processes)
#
#		feature_mat = np.zeros( (len(self.docs), num_features) )
#		list_of_args = list()
#		num_docs = len(self.docs) / self.processes
#		for pid in xrange(self.processes):
#			list_of_args.append( (self.docs, seeds, pid * num_docs, (pid + 1) * num_docs, feature_mat) )
#		pool.map(compute, list_of_args, chunksize=1)
#
#		print feature_mat
#		return feature_mat

class FastSeedKumarCONFIRM(KumarCONFIRM):
	'''
	When forming seeds from the clustering results, uses only a sample of documents to
		aggregate so that it goes fast
	'''

	def __init__(self, docs, min_docs=20, perc_docs=.1, **kwargs):
		super(FastSeedKumarCONFIRM, self).__init__(docs, **kwargs)
		self.min_docs = min_docs
		self.perc_docs = perc_docs
	
	def _form_seeds(self):
		seeds = list()
		for cluster in self.clusters:
			if not cluster.members:
				continue
			num_docs = int(len(cluster.members) * self.perc_docs)
			num_docs = max(num_docs, self.min_docs)
			num_docs = min(num_docs, len(cluster.members))
			docs = random.sample(cluster.members, num_docs)
			seed = docs[0].copy()
			for _doc in docs[1:]:
				seed.aggregate(_doc)
			seed.final_prune()
			seeds.append(seed)
		return seeds


#class BestKumarCONFIRM(RandomSeedsKumarCONFIRM, FastSeedKumarCONFIRM):
class BestKumarCONFIRM(KumarCONFIRM):
	pass

class PipelineCONFIRM(cluster.BaseCONFIRM):
	
	REJECT = -1

	def __init__(self, docs, init_subset=2000, min_membership=5, iterations=2, 
				num_initial_seeds=10, num_seeds=10, z_threshold=-1, 
				use_labels=True, **kwargs):
		self.docs = docs
		self.init_subset = init_subset
		self.min_cluster_membership = min_membership
		self.iterations = iterations
		self.num_initial_seeds = num_initial_seeds
		self.num_seeds = num_seeds
		self.z_threshold = z_threshold
		self.use_labels = use_labels

	def cluster(self):
		if self.use_labels:
			label_to_list = collections.defaultdict(list)
			for _doc in self.docs[:self.init_subset]:
				_doc._load_check()
				label = _doc.label
				label_to_list[label].append(_doc)
			clusters = list()
			for label in label_to_list:
				clusters.append(cluster.Cluster(label_to_list[label], label_to_list[label][0], label))
		else:
			kumarConfirm = BestKumarCONFIRM(self.docs[:self.init_subset], self.iterations, 
											self.num_initial_seeds, self.num_seeds)
			kumarConfirm.cluster()
			clusters = kumarConfirm.clusters

		clusters = self.preprocess_clusters(clusters)

		# greedy clustering with reject option based on thresholds
		self.rejected_docs = list()
		# clear assignments
		for cluster in clusters:
			cluster.members = list()
		for _doc in self.docs:
			cluster = self.nearest_cluster(clusters, _doc)
			if cluster != self.REJECT:
				cluster.members.append(_doc)
			else:
				self.rejected_docs.append(_doc)
		clusters = filter(lambda cluster: cluster.members, clusters)
		self.clusters = clusters
		return clusters

	def nearest_cluster(self, clusters, _doc):
		sim_scores = list()
		#z_scores = list()
		for cluster in clusters:
			sim_score = self.cluster_doc_similarity(cluster, _doc)
			#z_score = (sim_score - cluster.mean) / cluster.stddev
			sim_scores.append(sim_score)
			#z_scores.append(z_score)
		cluster = clusters[utils.argmax(sim_scores)]
		sim_score = max(sim_scores)
		z_score = (sim_score - cluster.mean) / cluster.stddev
		if z_score > self.z_threshold:
			return cluster
		else:
			return self.REJECT

	def print_reject_analysis(self):
		if self.rejected_docs:
			reject_cluster = cluster.Cluster(self.rejected_docs, center=self.rejected_docs[0])
			self.clusters.append(reject_cluster)
			print "REJECT Analysis"
		else:
			print "No rejected documents"

		analyzer = metric.KnownClusterAnalyzer(self)
		analyzer.print_general_info()
		analyzer.print_histogram_info()
		analyzer.print_label_conf_mat()
		analyzer.print_label_cluster_mat()
		analyzer.print_label_info()
		analyzer.print_metric_info()

		if self.rejected_docs:
			print "END REJECT Analysis"
			self.clusters.remove(reject_cluster)


	def preprocess_clusters(self, clusters):
		'''
		Forms clusters that will be fixed for greedy assignment of the remaining docs
		'''
		# remove too small clusters
		clusters = filter(lambda cluster: len(cluster.members) >= self.min_cluster_membership, clusters)

		self._form_prototypes(clusters)
		#self._push_apart_prototypes(clusters)
		self._compute_feature_weights(clusters)
		self._compute_sim_scores(clusters)
		self._estimate_cluster_thresholds(clusters)
		return clusters

	def cluster_doc_similarity(self, cluster, _doc):
		sim_vec = cluster.center.global_region_sim(_doc)
		return cluster.network.wavg(sim_vec)

	def _compute_sim_scores(self, clusters):
		'''
		Computes the sim scores for all docs to their cluster center
		'''
		for cluster in clusters:
			cluster.sim_scores = list()
			for _doc in cluster.members:
				cluster.sim_scores.append(self.cluster_doc_similarity(cluster, _doc))

	def _compute_feature_weights(self, clusters):
		'''
		Standard ML problem.  Take the data and compute weights (hopefully to generalize)
		'''
		for cluster in clusters:
			weights = None
			for _doc in cluster.members:
				sim_vec = cluster.center.global_region_sim(_doc)
				if weights is None:
					weights = sim_vec[:]
				else:
					for x in xrange(len(weights)):
						weights[x] += sim_vec[x]
			cluster.network = network.WeightedAverageNetwork(len(weights), weights, default_lr=0)

	# another idea for this is to use not just the positive scores, but to use the negative ones
	#  and use 1D logistic regression.  Of course that presupposes that all clusters are separate
	#  types
	def _estimate_cluster_thresholds(self, clusters):
		'''
		Uses the cluster sim scores to estimate cluster thresholds
		'''
		for cluster in clusters:
			#median = utils.median(cluster.sim_scores)
			mean = utils.avg(cluster.sim_scores)
			stddev = utils.stddev(cluster.sim_scores, mean=mean)
			if stddev == 0:
				print cluster.sim_scores
			cluster.mean = mean
			cluster.stddev = stddev

	# we might want to do merging at this point if things are too similar
	def _push_apart_prototypes(self, clusters):
		for cluster1 in clusters:
			for cluster2 in clusters:
				if cluster1 is cluster2:
					continue
				cluster1.center.push_away(cluster2.center)

	def _form_prototypes(self, clusters):
		'''
		Aggregates all cluster members together in order to create cluster centers
		'''
		for cluster in clusters:
			prototype = cluster.members[0].copy()
			for _doc in cluster.members[1:]:
				prototype.aggregate(_doc)
			prototype.final_prune()
			cluster.center = prototype
		
		

