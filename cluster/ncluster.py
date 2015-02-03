
import os
import datetime
import metric
import cluster
import network
import numpy as np
import scipy.spatial.distance
import sklearn.ensemble
import sklearn.cluster
import sklearn.linear_model
import sklearn.metrics
import collections
import utils
import random
import math
import multiprocessing
from constants import *

def remove_duplicate_features(data_matrix, diff=0.01):
	to_remove = list()
	for col in xrange(data_matrix.shape[1]):
		if col in to_remove:
			continue
		for col2 in xrange(col + 1, data_matrix.shape[1]):
			if col2 in to_remove:
				continue
			dist = scipy.spatial.distance.cityblock(data_matrix[:,col], data_matrix[:,col2])  
			norm_dist = dist / float(data_matrix.shape[0])
			#print "%d, %d: %.3f, %.3f" % (col, col2, dist, norm_dist)
			if norm_dist < diff:
				to_remove.append(col2)
	print "Removing %d/%d features" % (len(to_remove), data_matrix.shape[1])
	return np.delete(data_matrix, to_remove, axis=1)

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
		silhouette_samples = sklearn.metrics.silhouette_samples(dist_matrix, assignments, metric='precomputed')
		num_positive = (silhouette_samples > 0).sum()
		print "%d: %.3f %.3f%%" % (num_clusters, silhouette, num_positive * 100.0)
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
		#print num_clusters
		sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters , affinity="precomputed",
												assign_labels="discretize")
		assignments = sc.fit_predict(affinity_matrix)
		silhouette = sklearn.metrics.silhouette_score(dist_matrix, assignments, metric='precomputed')
		silhouette_samples = sklearn.metrics.silhouette_samples(dist_matrix, assignments, metric='precomputed')
		num_positive = (silhouette_samples > 0).sum()
		all_assignments[num_clusters] = (assignments, silhouette, num_positive)
	
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
	map(lambda _cluster: _cluster.set_label(), clusters)
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
	
	def __init__(self, docs, iterations=2, num_initial_seeds=10, num_seeds=10, cluster_range=(2,4), **kwargs):
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
			assignments, silhouette, num_positive = all_assignments[num_clusters]
			self.clusters = form_clusters(instances, assignments)
			analyzer = metric.KnownClusterAnalyzer(self)
			accuracy = analyzer.accuracy()
			v_measure = analyzer.v_measure()
			print "%d: %2.1f%% %.3f %.3f %.3f%%" % (num_clusters, accuracy * 100, v_measure, silhouette, 100.0 * num_positive / len(self.docs))
			if accuracy > best_acc:
				best_acc = accuracy
				best_assignments = assignments
		print 
		print "Best Acc: ", best_acc
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
			reduced = remove_duplicate_features(data_matrix, DUP_THRESH)
			if iteration < (self.iterations - 1):
				self.clusters = self.kumar_cluster_cv_cheat(data_matrix, self.docs, self.cluster_range)
				#self.clusters = kumar_cluster(data_matrix, self.docs, self.num_seeds)
				seeds = self._form_seeds()
			else:
				print self.cluster_range
				self.clusters = self.kumar_cluster_cv_cheat(data_matrix, self.docs, self.cluster_range)
				print
				print "Reduced"
				print
				self.clusters = self.kumar_cluster_cv_cheat(reduced, self.docs, self.cluster_range)
				#self.clusters = kumar_cluster(data_matrix, self.docs, self.num_seeds)
			print "In KumarCONFIRM"
			self.print_analysis()
				

	def _choose_initial_seeds(self):
		return self.docs[0:self.num_initial_seeds]

	def _form_seeds(self):
		seeds = list()
		for _cluster in self.clusters:
			if not _cluster.members:
				continue
			seed = _cluster.members[0].copy()
			for _doc in _cluster.members[1:]:
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

class FeatureExtractor(object):
	
	def __init__(self, docs):
		self.docs = docs
		random.shuffle(self.docs)

		self.all_labels = map(lambda _doc: _doc.label, self.docs)
		self.labels = list(set(self.all_labels))
		self.mapping = {label: self.labels.index(label) for label in self.labels}
		self.true_labels = map(lambda _doc: self.mapping[_doc.label], self.docs)

	def extract_type(self, outdir, num_seeds, perc_types):
		print
		print "extract_type()"
		print
		try:
			os.makedirs(outdir)
		except:
			pass

		np.save(os.path.join(outdir, "labels.npy"), self.true_labels)

		perc_types.sort()
		num_types = len(self.labels)
		num_types_to_try = list()
		for perc_type in perc_types:
			num = int(math.ceil(num_types * perc_type))
			if num not in num_types_to_try:
				num_types_to_try.append(num)

		type_histogram = collections.Counter(self.all_labels)
		biggest_types = map(lambda tup: tup[0], type_histogram.most_common(num_types))

		docs_by_type = collections.defaultdict(list)
		for _doc in self.docs:
			docs_by_type[_doc.label].append(_doc)

		print "num_types_to_try", num_types_to_try
		print
		for num in num_types_to_try:
			print num
			types = biggest_types[:num]
			forms_per_type = num_seeds / num
			extra = num_seeds % num

			seeds = list()
			for x, _type in enumerate(types):
				num_to_sample = forms_per_type
				if x < extra:
					num_to_sample += 1
				if num_to_sample > len(docs_by_type[_type]):
					seeds += docs_by_type[_type]
				else:
					seeds += random.sample(docs_by_type[_type], num_to_sample)

			#assert num_seeds == len(seeds)

			mat = self._compute_features(seeds)[0]
			np.save(os.path.join(outdir, "type_%d_%d.npy" % (num, num_seeds)), mat)

	def extract_random(self, outdir, amounts):
		print
		print "extract_random()"
		print
		try:
			os.makedirs(outdir)
		except:
			pass
		np.save(os.path.join(outdir, "labels.npy"), self.true_labels)

		amounts.sort()
		max_amount = amounts[-1]
		seeds = random.sample(self.docs, max_amount)
		mat, end_posses = self._compute_features(seeds)

		print
		print "Saving matrices"
		print
		for amount in amounts:
			print amount
			end_pos = end_posses[amount]
			sub_mat = mat[:,:end_pos]
			np.save(os.path.join(outdir, "rand_%d.npy" % amount), sub_mat)

	def _compute_features(self, seeds):
		num_features = 0
		_doc = self.docs[0]
		vectors = list()
		for seed in seeds:
			vectors.append(seed.match_vector(_doc))
		num_features = sum(map(len, vectors))
		feature_mat = np.zeros( (len(self.docs), num_features) )
		end_poses = list()
		for x, _doc in enumerate(self.docs):
			if x % 20 == 0:
				print "\t%d/%d (%.2f%%) Documents Extracted" % (x, len(self.docs), 100. * x / len(self.docs))
			offset = 0
			for seed in seeds:
				vector = seed.match_vector(_doc)
				feature_mat[x,offset:offset + len(vector)] = vector
				offset += len(vector)
				end_poses.append(offset)
		return feature_mat, end_poses
		


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
	
	def __init__(self, docs, num_seeds=-1, num_per_seed=1, **kwargs):
		super(SemiSupervisedKumarCONFIRM, self).__init__(docs, **kwargs)
		self.num_per_seed = num_per_seed
		self.num_seeds = num_seeds

	def _choose_initial_seeds(self):
		dseeds = collections.defaultdict(list)
		num_labels = len(set(map(lambda _doc: _doc.label, self.docs)))
		if self.num_seeds != -1:
			self.num_per_seed = self.num_seeds / num_labels + 1 
		for _doc in self.docs:
			_doc._load_check()
			label = _doc.label
			if len(dseeds[label]) < self.num_per_seed:
				dseeds[label].append(_doc)
		seeds = list()
		for x in xrange(self.num_per_seed):
			for label in dseeds:
				seeds.append(dseeds[label][x])
		if self.num_seeds != -1:
			seeds = seeds[:self.num_seeds]
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
		for _cluster in self.clusters:
			if not _cluster.members:
				continue
			num_docs = int(len(_cluster.members) * self.perc_docs)
			num_docs = max(num_docs, self.min_docs)
			num_docs = min(num_docs, len(_cluster.members))
			docs = random.sample(_cluster.members, num_docs)
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
				use_labels=False, use_ss=False, num_per_seed=1, **kwargs):
		self.docs = docs
		self.init_subset = init_subset
		self.min_cluster_membership = min_membership
		self.iterations = iterations
		self.num_initial_seeds = num_initial_seeds
		self.num_seeds = num_seeds
		self.z_threshold = z_threshold
		self.use_labels = use_labels
		self.topN = 3
		self.use_ss = use_ss
		self.num_per_seed = num_per_seed
	
	def cluster_doc_similarity(self, cluster, _doc):
		sim_vec = cluster.center.global_region_sim(_doc)
		return cluster.network.wavg(sim_vec)

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

	def push_apart_clusters_topN(self, N):
		sim_mat = self.get_cluster_sim_mat()

		# ordered pairs, lowest cluster idx is always first
		pairs = set()
		for clust1, row in enumerate(sim_mat):
			score_idx = zip(row, range(len(row)))
			score_idx.sort(key=lambda tup: tup[0], reverse=True)
			for i in xrange(min(N, len(score_idx) - 1)):
				# i + 1 to avoid the first entry, which should be clust1
				clust2 = score_idx[i + 1][1]
				if clust1 < clust2:
					pairs.add( (clust1, clust2) )
				else:
					pairs.add( (clust2, clust1) )

		for clust1, clust2 in pairs:
			self.clusters[clust1].center.push_away(self.clusters[clust2].center)

	def greedy(self, clusters):
		for _cluster in clusters:
			_cluster.members = list()

		# greedy clustering
		for _doc in self.docs:
			_cluster = self.nearest_cluster(clusters, _doc)
			#if _cluster != self.REJECT:
			_cluster.members.append(_doc)
		return clusters

	def get_affinity_matrix(self):
		if self.use_ss:
			kumarConfirm = SemiSupervisedKumarCONFIRM(self.docs[:self.init_subset], num_per_seed=self.num_per_seed,
											iterations=self.iterations, num_initial_seeds=self.num_initial_seeds, 
											num_seeds=self.num_seeds)
			seeds = kumarConfirm._choose_initial_seeds()

		else:
			kumarConfirm = KumarCONFIRM(self.docs[:self.init_subset], self.iterations, 
											self.num_initial_seeds, self.num_seeds)
			seeds = kumarConfirm._choose_initial_seeds()
		print "Num Seeds: ", len(seeds)

		data_matrix = self._compute_features(seeds)
		if REMOVE_DUP_FEATURES:
			data_matrix = remove_duplicate_features(data_matrix, DUP_THRESH)
		random_matrix = compute_random_matrix(data_matrix)
		rf = train_random_forest(data_matrix, random_matrix)
		sim_matrix = compute_sim_mat(data_matrix, rf)
		return sim_matrix

	def cluster_bootstrap(self):
		outdir = os.path.join(FEATURES_OUTDIR, str(datetime.datetime.now()).replace(' ', '_') + "_".join(sys.argv[1:]))
		try:
			os.makedirs(outdir)
		except:
			pass
		all_labels = list(set(map(lambda _doc: _doc.label, self.docs)))
		mapping = {label: all_labels.index(label) for label in all_labels}
		true_labels = np.array(map(lambda _doc: mapping[_doc.label], self.docs))
		np.save(os.path.join(outdir, 'true_labels.npy'), true_labels)

		affinity_matrix = self.get_affinity_matrix()
		for num_clusters in NUM_CLUSTERS:
			try:
				os.makedirs(os.path.join(outdir, str(num_clusters)))
			except:
				pass
			assignments = spectral_cluster(affinity_matrix, num_clusters)
			clusters = form_clusters(self.docs, assignments)
			self.clusters = filter(lambda cluster: cluster.members, clusters)

			analyzer = metric.KnownClusterAnalyzer(self)
			silhouette = sklearn.metrics.silhouette_score(1 - affinity_matrix, assignments, metric='precomputed')
			analyzer.print_summary(num_clusters, self.num_seeds, prefix="init", sil=silhouette)
			analyzer.print_general_info()
			analyzer.print_label_conf_mat()
			analyzer.print_label_cluster_mat()

			clusters = filter(lambda _cluster: len(_cluster.members) >= self.min_cluster_membership, clusters)
			self._form_prototypes(clusters)
			features = self._compute_features(map(lambda _cluster: _cluster.center, clusters))
			labels = list()
			for _doc in self.docs:
				for x,_cluster in enumerate(clusters):
					if _doc in _cluster.members:
						labels.append(x)
						break
			labels = np.array(labels)
			np.save(os.path.join(outdir, str(num_clusters), "features.npy"), features)
			np.save(os.path.join(outdir, str(num_clusters), "bootstrapped_labels.npy"), labels)

			classifier = sklearn.linear_model.LogisticRegression()
			classifier.fit(features, labels)
			print classifier.coef_
			assignments = classifier.predict(features)
			self.clusters = form_clusters(self.docs, assignments)
			np.save(os.path.join(outdir, str(num_clusters), "predicted_labels.npy"), assignments)

			analyzer = metric.KnownClusterAnalyzer(self)
			analyzer.print_summary(num_clusters, self.num_seeds, prefix="final")
			self.print_reject_analysis()

		

	def cluster2(self):
		affinity_matrix = self.get_affinity_matrix()
		for num_clusters in NUM_CLUSTERS:
			assignments = spectral_cluster(affinity_matrix, num_clusters)
			clusters = form_clusters(self.docs, assignments)
			self.clusters = filter(lambda cluster: cluster.members, clusters)

			analyzer = metric.KnownClusterAnalyzer(self)
			silhouette = sklearn.metrics.silhouette_score(1 - affinity_matrix, assignments, metric='precomputed')
			analyzer.print_summary(num_clusters, self.num_seeds, prefix="init", sil=silhouette)
			analyzer.print_general_info()
			analyzer.print_label_conf_mat()
			analyzer.print_label_cluster_mat()

			clusters = self.preprocess_clusters(clusters, do_prototypes=True)
			clusters = self.greedy(clusters)
			self.clusters = filter(lambda cluster: cluster.members, clusters)
			analyzer = metric.KnownClusterAnalyzer(self)
			analyzer.print_summary(num_clusters, self.num_seeds, prefix="final")
			self.print_reject_analysis()
			

	def cluster(self):
		if self.use_labels:
			label_to_list = collections.defaultdict(list)
			for _doc in self.docs[:self.init_subset]:
				_doc._load_check()
				label = _doc.label
				label_to_list[label].append(_doc)
			clusters = list()
			for label in label_to_list:
				clusters.append(cluster.Cluster(label_to_list[label], label_to_list[label][0]))
		elif self.use_ss:
			kumarConfirm = SemiSupervisedKumarCONFIRM(self.docs[:self.init_subset], num_per_seed=self.num_per_seed,
											iterations=self.iterations, num_initial_seeds=self.num_initial_seeds, 
											num_seeds=self.num_seeds)
			kumarConfirm.cluster()
			clusters = kumarConfirm.clusters

		else:
			kumarConfirm = KumarCONFIRM(self.docs[:self.init_subset], self.iterations, 
											self.num_initial_seeds, self.num_seeds)
			kumarConfirm.cluster()
			clusters = kumarConfirm.clusters

		for _cluster in clusters:
			_cluster.original_members = _cluster.members[:]

		self.clusters = clusters


		for _i in xrange(1):
			print
			print "*" * 30
			print "ITERATION %d" % _i
			print "*" * 30
			print
			if _i:
				self.push_apart_clusters_topN(self.topN)
			self.clusters = self.preprocess_clusters(self.clusters, do_prototypes=(_i == 0))

			for x, _cluster in enumerate(self.clusters):
				print "\nCluster %d: %s, size: %d" % (x, _cluster.label, len(_cluster.original_members))
				if hasattr(_cluster, 'network'):
					print "\t" + (" ".join(map(lambda w: "%.3f" % w, _cluster.network.weights)))
				if hasattr(_cluster, 'mean'):
					print "\t%.2f\t%.2f" % (_cluster.mean, _cluster.stddev)
			#self.print_reject_analysis(draw=True)

			# greedy clustering with reject option based on thresholds
			#self.rejected_docs = list()

			# clear assignments
			for _cluster in self.clusters:
				_cluster.members = list()

			# greedy clustering
			for _doc in self.docs:
				_cluster = self.nearest_cluster(clusters, _doc)
				#if _cluster != self.REJECT:
				_cluster.members.append(_doc)
				#else:
				#	self.rejected_docs.append(_doc)
			print "In PipelineKumarCONFIRM"
			self.print_full_analysis(draw=True)

		self.clusters = filter(lambda _cluster: _cluster.members, self.clusters)
		return self.clusters

	def nearest_cluster(self, clusters, _doc):
		sim_scores = list()
		#z_scores = list()
		for _cluster in clusters:
			sim_score = self.cluster_doc_similarity(_cluster, _doc)
			#z_score = (sim_score - _cluster.mean) / _cluster.stddev
			sim_scores.append(sim_score)
			#z_scores.append(z_score)
		_cluster = clusters[utils.argmax(sim_scores)]
		return _cluster
		#sim_score = max(sim_scores)
		#z_score = (sim_score - _cluster.mean) / _cluster.stddev
		#if z_score > self.z_threshold:
		#	return _cluster
		#else:
		#	return self.REJECT

	def print_reject_analysis(self, draw=False):
		#if self.rejected_docs:
		#	reject_cluster = cluster.Cluster(self.rejected_docs, center=self.rejected_docs[0])
		#	self.clusters.append(reject_cluster)
		#	print "REJECT Analysis"
		#else:
		#	print "No rejected documents"

		analyzer = metric.KnownClusterAnalyzer(self)
		analyzer.print_general_info()
		analyzer.print_histogram_info()
		analyzer.print_label_conf_mat()
		analyzer.print_label_cluster_mat()
		analyzer.print_label_info()
		analyzer.print_metric_info()
		if draw:
			analyzer.draw_centers()

		#if self.rejected_docs:
		#	print "END REJECT Analysis"
		#	self.clusters.remove(reject_cluster)

	def print_full_analysis(self, draw=False):
		analyzer = metric.KnownClusterAnalyzer(self)
		analyzer.print_all()
		if draw:
			analyzer.draw_centers()


	def preprocess_clusters(self, clusters, do_prototypes=True):
		'''
		Forms clusters that will be fixed for greedy assignment of the remaining docs
		'''
		# remove too small clusters
		clusters = filter(lambda _cluster: len(_cluster.members) >= self.min_cluster_membership, clusters)

		if do_prototypes:
			self._form_prototypes(clusters)
		#self._push_apart_prototypes(clusters)
		self._compute_feature_weights(clusters)
		#self._compute_sim_scores(clusters)
		#self._estimate_cluster_thresholds(clusters)
		return clusters

	def cluster_doc_similarity(self, _cluster, _doc):
		sim_vec = _cluster.center.global_region_sim(_doc)
		return _cluster.network.wavg(sim_vec)

	def _compute_sim_scores(self, clusters):
		'''
		Computes the sim scores for all docs to their cluster center
		'''
		for _cluster in clusters:
			_cluster.sim_scores = list()
			for _doc in _cluster.original_members:
				_cluster.sim_scores.append(self.cluster_doc_similarity(_cluster, _doc))

	def _compute_feature_weights(self, clusters):
		'''
		Standard ML problem.  Take the data and compute weights (hopefully to generalize)
		'''
		for _cluster in clusters:
			weights = None
			#for _doc in _cluster.original_members:
			for _doc in _cluster.members:
				sim_vec = _cluster.center.global_region_sim(_doc)
				if weights is None:
					weights = sim_vec[:]
				else:
					for x in xrange(len(weights)):
						weights[x] += sim_vec[x]
			_cluster.network = network.WeightedAverageNetwork(len(weights), weights, default_lr=0)

	# another idea for this is to use not just the positive scores, but to use the negative ones
	#  and use 1D logistic regression.  Of course that presupposes that all clusters are separate
	#  types
	def _estimate_cluster_thresholds(self, clusters):
		'''
		Uses the cluster sim scores to estimate cluster thresholds
		'''
		for _cluster in clusters:
			#median = utils.median(_cluster.sim_scores)
			mean = utils.avg(_cluster.sim_scores)
			stddev = utils.stddev(_cluster.sim_scores, mean=mean)
			if stddev == 0:
				print _cluster.sim_scores
			_cluster.mean = mean
			_cluster.stddev = stddev

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
		for _cluster in clusters:
			prototype = _cluster.members[0].copy()
			for _doc in _cluster.members[1:]:
				prototype.aggregate(_doc)
			prototype.final_prune()
			_cluster.center = prototype

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
		return feature_mat
		
		


