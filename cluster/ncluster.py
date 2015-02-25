
import os
import datetime
import metric
import cluster
import network
import selector
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

class MockCenter:
	pass

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
	#print "Removing %d/%d features" % (len(to_remove), data_matrix.shape[1])
	return np.delete(data_matrix, to_remove, axis=1)


def compute_structured_random_matrix(data_matrix, stay_prob=0.5):
	#print "Constructing Structured Random Training Set"

	rand_shape = (int(data_matrix.shape[0] * SIZE_OF_RANDOM_DATA), data_matrix.shape[1])
	rand_mat = np.zeros(rand_shape)

	# these are the indices of the rows we sample from
	row_choices = xrange(data_matrix.shape[0])

	# column indices for both matrices
	col_idxs = xrange(rand_mat.shape[1])
	for row in xrange(rand_mat.shape[0]):

		# pick a row idx from data_matrix
		sampled_row = random.choice(row_choices)

		# always fill in the columns in random order
		random.shuffle(col_idxs)
		for col in col_idxs:
			
			# fill in the value using the corresponding column of the sampled row
			rand_mat[row, col] = data_matrix[sampled_row, col]

			# retain the same sampled row for the next column with prob $stay_prob
			if random.random() > stay_prob:
				sampled_row = random.choice(row_choices)

	#print "Done\n"
	return rand_mat


def compute_random_matrix(data_matrix):
	#print "Constructing Random Training Set"

	rand_shape = (int(data_matrix.shape[0] * SIZE_OF_RANDOM_DATA), data_matrix.shape[1])
	rand_mat = np.zeros(rand_shape)
	for col in xrange(rand_mat.shape[1]):
		vals = data_matrix[:,col]
		for row in xrange(rand_mat.shape[0]):
			rand_mat[row, col] = np.random.choice(vals)

	#print "Done\n"
	return rand_mat


def train_random_forest(real_data, fake_data):
	#print "Training Random Forest"

	#print "Num features:", real_data.shape[1]
	num_tree_features = FUNCTION_NUM_FEATURES(real_data.shape[1])
	#print "Computed Tree features:", num_tree_features
	rf = sklearn.ensemble.RandomForestClassifier(n_estimators=NUM_TREES, max_features=num_tree_features,
												bootstrap=False, n_jobs=RF_THREADS)
	combined_data = np.concatenate( (real_data, fake_data) )
	labels = np.concatenate( (np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])) )
	rf.fit(combined_data, labels)

	#print "Done\n"
	return rf

def compute_sim_mat(data_matrix, random_forest):
	leaf_nodes = random_forest.apply(data_matrix)
	sim_mat = scipy.spatial.distance.pdist(leaf_nodes, "hamming")
	sim_mat = scipy.spatial.distance.squareform(sim_mat)
	sim_mat = 1 - sim_mat

	return sim_mat

def spectral_cluster(affinity_matrix, num_clusters):
	#print "Performing Spectral Clustering"

	sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters , affinity="precomputed",
											assign_labels="discretize")
	assignments = sc.fit_predict(affinity_matrix)

	#print "Done\n"
	return assignments


def form_clusters(instances, assignments):
	'''
	Takes a list of instances and assignments and returns 
		a list of Cluster objects with Mocked centers
	'''
	cluster_map = dict()
	m = MockCenter()
	m.label = None
	for x in xrange(assignments.max() + 1):
		cluster_map[x] = cluster.Cluster(list(), m, x)
	for instance, assignment in zip(instances, assignments):
		cluster_map[assignment].members.append(instance)
	clusters = cluster_map.values()
	clusters = filter(lambda c: len(c.members), clusters)
	map(lambda _cluster: _cluster.set_label(), clusters)
	return clusters


def form_clusters_alt(instances, l_idx):
	'''
		instances - list of clustered things
		l_idx - list of lists of indices into instances
			e.g. [ [1, 3, 5], [0, 2, 4] ]
	'''
	clusters = list()
	m = MockCenter()
	m.label = None
	for x, l in enumerate(l_idx):
		_cluster = cluster.Cluster(list(), m)
		for idx in l:
			_cluster.members.append(instances[idx])
		clusters.append(_cluster)
	clusters = filter(lambda c: len(c.members), clusters)
	map(lambda _cluster: _cluster.set_label(), clusters)
	return clusters
	

def set_cluster_center(_cluster):
	center = _cluster.members[0].copy()
	for _doc in _cluster.members[1:]:
		center.aggregate(_doc)
	center.final_prune()
	_cluster.center = center


def set_cluster_centers(clusters):
	clusters = filter(lambda c: len(c.members), clusters)
	for _cluster in clusters:
		set_cluster_center(_cluster)
	

# Note that there are many ways we could do this
def cluster_dist_mat(_cluster, feature_type='match', dist_metric='euclidean'):
	features = extract_features(_cluster.members, [_cluster.center], feature_type)[0]
	
	if dist_metric == 'rf':
		rand_mat = compute_random_matrix(features)
		rf = train_random_forest(features, rand_mat)
		sim_mat = compute_sim_mat(features, rf)
		dists = 1 - sim_mat
	else:
		dists = scipy.spatial.distance.pdist(features, 'euclidean')
		dists = scipy.spatial.distance.squareform(dists)
	return dists

def split_clusters(clusters, min_size, feature_type='match', dist_metric='euclidean'):
	#set_cluster_centers(clusters)
	split_clusters = map(lambda _cluster: split_cluster(_cluster, min_size, feature_type, dist_metric), clusters)
	return utils.flatten(split_clusters)

def split_cluster(_cluster, min_size, feature_type='match', dist_metric='euclidean'):
	'''
	Splits a cluster using Logan's OPTICS
		Returns a list of resulting clusters (perhaps just the original)
	'''
	if not _cluster.center:
		set_cluster_center(_cluster)

	dist_mat = cluster_dist_mat(_cluster, feature_type, dist_metric)
	reachabilities = selector.OPTICS(dist_mat, min_size)
	indices = selector.separateClusters(reachabilities, min_size)

	# comes back as selector.dataPoint classes
	indices = map(lambda l: map(lambda dp: dp._id, l), indices)
	#if len(indices) == 1:
	#	return [_cluster]
	#print indices
	clusters = form_clusters_alt(_cluster.members, indices)

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


def calc_num_features(seeds, feature_type='match'):
	'''
	Calculates the total number of features obtained by matching
		against all seeds
	'''
	vectors = list()
	_doc = seeds[0]
	for seed in seeds:
		vectors.append(_extract_features(_doc, seed, feature_type))
	return sum(map(len, vectors))


def extract_matching_features_rand_seeds(docs, amounts):
	amounts.sort()
	max_amount = amounts[-1]
	seeds = random.sample(docs, max_amount)
	mat, end_posses = extract_features(docs, seeds, 'match')

	mats = list()
	for amount in amounts:
		end_pos = end_posses[amount]
		sub_mat = mat[:,:end_pos]
		mats.append(sub_mat)
	
	return mats

def _extract_features(_doc, seed, feature_type='match'):
	if feature_type == 'match':
		vector = seed.match_vector(_doc)
	elif feature_type == 'sim':
		vector = seed.global_region_sim(_doc)
	else:
		raise Exception("Unknown feature extraction type %s" % repr(feature_type))
	return vector


def extract_features(docs, seeds, feature_type='match',  _print=True):
	'''
	Takes all docs and matches them against all seeds to produce
		a matrix of features.
		offsets[n] is the col index in feature_mat of the end of the nth seed's features
	'''
	num_docs = len(docs)
	num_features = calc_num_features(seeds, feature_type)

	feature_mat = np.zeros( (num_docs, num_features) )
	offsets = list()
	offsets.append(0)
	for x, _doc in enumerate(docs):
		if x % 20 == 0 and _print:
			pass
			#print "\t%d/%d (%.2f%%) Documents Extracted" % (x, num_docs, 100. * x / num_docs)
		offset = 0
		for seed in seeds:
			vector = _extract_features(_doc, seed, feature_type)
			feature_mat[x,offset:offset + len(vector)] = vector
			offset += len(vector)
			if x == 0:
				offsets.append(offset)
	return feature_mat, offsets


def extract_matching_features_rand_seeds(docs, amounts):
	amounts.sort()
	max_amount = amounts[-1]
	seeds = random.sample(docs, max_amount)
	mat, end_posses = extract_features(docs, seeds, 'match')

	mats = list()
	for amount in amounts:
		end_pos = end_posses[amount]
		sub_mat = mat[:,:end_pos]
		mats.append(sub_mat)
	
	return mats


def extract_type(docs, num_seeds, perc_types):
	all_labels = map(lambda _doc: _doc.label, docs)
	labels = list(set(all_labels))
	mapping = {label: labels.index(label) for label in labels}
	true_labels = map(lambda _doc: mapping[_doc.label], docs)

	perc_types.sort()
	num_types = len(labels)
	num_types_to_try = list()
	for perc_type in perc_types:
		num = int(math.ceil(num_types * perc_type))
		if num not in num_types_to_try:
			num_types_to_try.append(num)

	type_histogram = collections.Counter(all_labels)
	biggest_types = map(lambda tup: tup[0], type_histogram.most_common(num_types))

	docs_by_type = collections.defaultdict(list)
	for _doc in docs:
		docs_by_type[_doc.label].append(_doc)

	mats = list()
	for num in num_types_to_try:
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

		mat = extract_matching_features(docs, seeds)[0]
		mats.append(mat)
	return mats


def print_cluster_analysis(clusters, misclustered=False):
	class Mock:
		pass
	m = Mock()
	m.get_clusters = lambda: clusters
	analyzer = metric.KnownClusterAnalyzer(m)
	analyzer.print_general_info()
	analyzer.print_label_conf_mat()
	analyzer.print_label_cluster_mat()
	analyzer.print_metric_info()

	#if misclustered:
		#analyzer.


def get_acc_v_measure(clusters):
	class Mock:
		pass
	m = Mock()
	m.get_clusters = lambda: clusters
	analyzer = metric.KnownClusterAnalyzer(m)
	acc = analyzer.accuracy()
	v = analyzer.v_measure()
	return acc, v
	

def test_features(clusters, min_size):
	#print "Original Clusters"
	#print_cluster_analysis(clusters)
	set_cluster_centers(clusters)
	for dist in ['rf', 'euclidean']:
		for _type in ['match', 'sim']:
			sclusters = split_clusters(clusters, min_size, _type, dist)
			acc, v = get_acc_v_measure(sclusters)
			print "%s_%s\t%d\t%.lf\t%.3f" % (_type, dist, len(sclusters), acc, v)
			#print_cluster_analysis(sclusters)


def do_cluster(docs, num_seeds, num_clusters):
	seeds = random.sample(docs, num_seeds)
	features = extract_features(docs, seeds)[0]
	clusters = kumar_cluster(features, docs, num_clusters)
	return clusters
	

def test_splitting(docs):
	print "\t".join(["Method", "Original_Num_Clusters", "Split_Num_Clusters", "Diff_Num_Clusters", "Original_Acc", 
					"Split_Acc", "Diff_Acc", "Original_V-measure", "Split_V-measure", "Diff_V-measure"])
	min_size = max(int(len(docs) * 0.005), 2)
	for k in NUM_CLUSTERS:
		num_seeds = 10
		clusters = do_cluster(docs, num_seeds, k)
		set_cluster_centers(clusters)
		o_acc, o_v = get_acc_v_measure(clusters)
		print "\t".join(["original", str(k), str(k), '0'] + 
			map(lambda x: "%.3f" % x, [o_acc, o_acc, 0, o_v, o_v, 0]))
		for dist in ['rf', 'euclidean']:
			for _type in ['match', 'sim']:
				sclusters = split_clusters(clusters, min_size, _type, dist)
				acc, v = get_acc_v_measure(sclusters)
				d_acc = acc - o_acc
				d_v = v - o_v
				k2 = len(sclusters)
				print "\t".join(["%s_%s" % (_type, dist), str(k), str(k2), str(k2 - k)] + 
					map(lambda x: "%.3f" % x, [o_acc, acc, d_acc, o_v, v, d_v]))


def all_cluster(docs, num_subset, num_initial_clusters, num_seeds, min_pts, outdir):
	try:
		os.makedirs(outdir)
	except:
		print "Could not create dir: ", outdir
	
	# do the initial clustering
	random.shuffle(docs)
	subset = docs[:num_subset]
	num_subset = len(subset)  # just in case num_subset > len(docs)
	initial_clusters = do_cluster(subset, num_seeds, num_initial_clusters)

	# metrics
	print "*" * 30
	print "Initial Clusters:"
	print "*" * 30
	print_cluster_analysis(initial_clusters)

	set_cluster_centers(initial_clusters)
	utils.save_obj(docs, os.path.join(outdir, "docs.pckl"))
	utils.save_obj(subset, os.path.join(outdir, "subset.pckl"))
	utils.save_obj(initial_clusters, os.path.join(outdir, "initial_clusters.pckl"))

	# split the initial clusters
	sclusters = split_clusters(initial_clusters, min_pts, 'match', 'rf')

	# metrics
	print "*" * 30
	print "Split Clusters:"
	print "*" * 30
	print_cluster_analysis(sclusters)

	set_cluster_centers(sclusters)
	utils.save_obj(sclusters, os.path.join(outdir, "split_clusters.pckl"))


	# get the features for final classification
	centers = map(lambda _cluster: _cluster.center, sclusters)
	features = extract_features(docs, centers)[0]
	training_labels = np.zeros(num_subset, dtype=np.int16)
	for x, _doc in enumerate(subset):
		for y,  _cluster in enumerate(sclusters):
			if _doc in _cluster.members:
				training_labels[x] = y
				break
	training_features = features[:num_subset,:]

	np.save(os.path.join(outdir, 'features.npy'), features)
	np.save(os.path.join(outdir, 'training_features.npy'), training_features)
	np.save(os.path.join(outdir, 'training_labels.npy'), training_labels)

	# train classifier
	rf = sklearn.ensemble.RandomForestClassifier(n_estimators=NUM_TREES, bootstrap=False, 
												n_jobs=RF_THREADS)
	rf.fit(training_features, training_labels)
	utils.save_obj(rf, os.path.join(outdir, "rf.pckl"))

	# do classification
	assignments = rf.predict(features)

	# create clusters
	final_clusters = form_clusters(docs, assignments)
	utils.save_obj(final_clusters, os.path.join(outdir, "final_clusters.pckl"))
				
	# metrics
	print "*" * 30
	print "Final Clusters:"
	print "*" * 30
	print_cluster_analysis(final_clusters, True)

	# summary compare
	for name, clusters in {"init": initial_clusters, "split": sclusters, "final": final_clusters}.iteritems():
		acc, v = get_acc_v_measure(clusters)
		k = len(clusters)
		print "%s\t%d\t%.4f\t%.4f" % (name, k, acc, v)


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
		
		


