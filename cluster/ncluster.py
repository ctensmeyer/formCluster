
import os
import metric
import cluster
import selector
import numpy as np
import scipy.spatial.distance
import sklearn.ensemble
import sklearn.cluster
import sklearn.linear_model
import sklearn.metrics
import utils
import random
from constants import *


class MockCenter:
	pass


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
												bootstrap=False, n_jobs=THREADS)
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

def spectral_cluster(affinity_matrix, num_clusters, distance='rf'):
	#print "Performing Spectral Clustering"

	if distance == 'rf':
		sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters , affinity="precomputed",
												assign_labels="discretize")
	else:
		sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters , affinity="rbf",
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
	return center


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

def cluster_dist_mats(clusters, feature_type='match', dist_metric='euclidean'):
	dist_mats = map(lambda _cluster: cluster_dist_mat(_cluster, feature_type, dist_metric), clusters)
	return dist_mats

def split_cluster(_cluster, dist_mat, args):
	'''
	Splits a cluster using Logan's OPTICS
		Returns a list of resulting clusters (perhaps just the original)
	'''
	if args.no_auto_minpts:
		min_size = args.minpts
	else:
		min_size = int(max(args.minpts, args.minpts_perc * len(_cluster.members)))
	reachabilities = selector.OPTICS(dist_mat, min_size)
	indices = selector.separateClusters(reachabilities, min_size)

	# comes back as selector.dataPoint classes
	indices = map(lambda l: map(lambda dp: dp._id, l), indices)
	clusters = form_clusters_alt(_cluster.members, indices)

	return clusters


def calc_num_features(seeds, feature_type='all'):
	'''
	Calculates the total number of features obtained by matching
		against all seeds
	'''
	vectors = list()
	_doc = seeds[0]
	for seed in seeds:
		vectors.append(_extract_features(_doc, seed, feature_type))
	return sum(map(len, vectors))


def _extract_features(_doc, seed, feature_type='all'):
	vector = seed.match_vector(_doc, feature_type)
	return vector


def extract_features(docs, seeds, feature_types='all',  _print=False):
	'''
	Takes all docs and matches them against all seeds to produce
		a matrix of features.
		offsets[n] is the col index in feature_mat of the end of the nth seed's features
	'''
	num_docs = len(docs)
	num_features = calc_num_features(seeds, feature_types)

	feature_mat = np.zeros( (num_docs, num_features) )
	offsets = list()
	offsets.append(0)
	for x, _doc in enumerate(docs):
		if x % 20 == 0 and _print:
			pass
			print "\t%d/%d (%.2f%%) Documents Extracted" % (x, num_docs, 100. * x / num_docs)
		offset = 0
		for seed in seeds:
			vector = _extract_features(_doc, seed, feature_types)
			feature_mat[x,offset:offset + len(vector)] = vector
			offset += len(vector)
			if x == 0:
				offsets.append(offset)
	return feature_mat, offsets



def print_cluster_analysis(clusters):
	class Mock:
		pass
	m = Mock()
	m.get_clusters = lambda: clusters
	analyzer = metric.KnownClusterAnalyzer(m)
	analyzer.print_general_info()
	analyzer.print_label_conf_mat()
	analyzer.print_label_cluster_mat()
	analyzer.print_metric_info()

def get_acc_v_measure(clusters):
	class Mock:
		pass
	m = Mock()
	m.get_clusters = lambda: clusters
	analyzer = metric.KnownClusterAnalyzer(m)
	acc = analyzer.accuracy()
	v = analyzer.v_measure()
	return acc, v
	

def sort_by_type(docs):
	docs_by_type = list()
	for _doc in docs:
		for sublist in docs_by_type:
			if sublist[0].label == _doc.label:
				sublist.append(_doc)
				break
		else:
			docs_by_type.append([_doc])
	docs_by_type.sort(key=len, reverse=True)
	return docs_by_type
	
def get_random_exemplars(docs, num_exemplars):
	largest_num_exemplars = max(num_exemplars)
	all_exemplars = docs[:largest_num_exemplars]
	assert (len(all_exemplars) == largest_num_exemplars)
	exemplar_index = { (num_e, 0): range(num_e) for num_e in num_exemplars }
	return all_exemplars, exemplar_index


def get_oracle_exemplars(docs, num_exemplars, num_types):
	all_exemplars = list()
	exemplar_index = dict()

	docs_by_type = sort_by_type(docs)
	num_total_types = len(docs_by_type)
	for num_e in num_exemplars:
		for num_t in num_types:
			cur_exemplars = list()
			num_each = num_e / num_t
			num_extra = num_e % num_t

			# grab exemplars for single execution of algorithm
			for type_idx in range(num_t):
				num_from_type = num_each
				if (type_idx < num_extra):
					num_from_type += 1
				cur_exemplars += docs_by_type[type_idx][:num_from_type]

			assert(len(cur_exemplars) == num_e)

			# add unique cur_exemplars to global list of exemplars
			for exemplar in cur_exemplars:
				if exemplar not in all_exemplars:
					all_exemplars.append(exemplar)

			# create index into global list for cur_exemplars
			cur_exemplar_index = list()
			for exemplar in cur_exemplars:
				idx = all_exemplars.index(exemplar)
				assert(idx >= 0)
				cur_exemplar_index.append(idx)
			exemplar_index[(num_e, num_t)] = cur_exemplar_index

	return all_exemplars, exemplar_index
	

def load_exemplars_from_file(docs, _file):
	exemplars = []
	for line in open(_file,'r').readlines():
		line = line.rstrip()
		for _doc in docs:
			if _doc._id == line:
				exemplars.append(_doc)
				break
	return exemplars
	

def get_exemplars(docs, num_exemplars, num_types, args):
	if args.exemplar_file:
		docs = load_exemplars_from_file(docs, args.exemplar_file)
	if num_types:
		return get_oracle_exemplars(docs, num_exemplars, num_types)
	else:
		return get_random_exemplars(docs, num_exemplars)

def calculate_feature_col_indices(index, offsets):
	cols = list()
	for idx in index:
		start = offsets[idx]
		stop = offsets[idx+1]
		cols += range(start,stop)
	cols.sort()
	return cols

def cluster_refinement(initial_clusters, args):
	set_cluster_centers(initial_clusters)
	dist_mats = cluster_dist_mats(initial_clusters)
	sclusters = utils.flatten(map(
		lambda _cluster, dist_mat: split_cluster(_cluster, dist_mat, args), 
			initial_clusters, dist_mats))

	return sclusters
	
def create_bootstrap_features(sclusters, docs, subset_size):
	set_cluster_centers(sclusters)
	prototypes = map(lambda _cluster: _cluster.center, sclusters)
	bootstrap_features = extract_features(docs, prototypes)[0]

	training_labels = np.zeros(subset_size, dtype=np.int16)
	subset = docs[:subset_size]
	for x, _doc in enumerate(subset):
		for y,  _cluster in enumerate(sclusters):
			if _doc in _cluster.members:
				training_labels[x] = y
				break
	training_features = bootstrap_features[:subset_size,:]
	
	return bootstrap_features, training_features, training_labels 

def rf_sim_mat(feature_mat):
	random_matrix = compute_random_matrix(feature_mat)
	rf = train_random_forest(feature_mat, random_matrix)
	sim_matrix = compute_sim_mat(feature_mat, rf)
	return sim_matrix

def euclidean_sim_mat(feature_mat):
	dists = scipy.spatial.distance.pdist(feature_mat, 'euclidean')
	dists = scipy.spatial.distance.squareform(dists)
	sim_matrix = np.exp(- np.square(dists))
	return sim_matrix

def calc_sim_matrix(feature_mat, distance):
	if distance == 'rf':
		return rf_sim_mat(feature_mat)
	elif distance == 'euclidean':
		return euclidean_sim_mat(feature_mat)

def bootstrap_cluster(sclusters, docs, subset_size):
	# Construct features for training and prediction
	bootstrap_features, training_features, training_labels = create_bootstrap_features(sclusters, docs, subset_size)

	# Train LR classifier and predict clusters for all of data
	lr = sklearn.linear_model.LogisticRegression(penalty='l1')
	lr.fit(training_features, training_labels)
	assignments = lr.predict(bootstrap_features)
	bootstrap_clusters = form_clusters(docs, assignments)

	return bootstrap_clusters 

def initial_cluster(mat, k, subset, distance):
	assignments = spectral_cluster(mat, k, distance)
	initial_clusters = form_clusters(subset, assignments)
	return initial_clusters 

def print_clusters(clusters, title, tag):
	print "%s\n%s\n%s" % ("*" * 30, "%s Clusters:" % title, "*" * 30)
	print_cluster_analysis(clusters)

	acc, v = get_acc_v_measure(clusters)
	num_clusters = len(clusters)
	print "tag K subset num_e num_t purity v K'"
	print "%s %.5f %.5f %d" % (tag, acc, v, num_clusters)
	

def confirm(docs, Ks, subset_sizes, num_exemplars, num_types, args):
	largest_subset_size = max(subset_sizes)
	largest_subset = docs[:largest_subset_size]

	# all_exemplars is a list of unique exemplars used across all specified parameter settings
	# of num_exemplars and num_types.  This allows feature extraction to be done once to save
	# redundant computation between experiments.
	# exemplar_index is a dictionary whose entry (num_e, num_t) is a list of indices into
	# all_exemplars.  When random exemplars are used, $num_t = 0 as a sentinel value
	all_exemplars, exemplar_index = get_exemplars(largest_subset, num_exemplars, num_types, args)

	# precompute features for all docs in a subset with all exemplars used
	feature_types = []
	if not args.no_all:
		feature_types.append('all')
	if args.text_only:
		feature_types.append('text')
	if args.rule_only:
		feature_types.append('rule')

	for feature_type in feature_types:
		all_feature_mat, exemplar_offsets = extract_features(largest_subset, all_exemplars, feature_type)

		for num_e, num_t in sorted(exemplar_index.keys()):
			
			# select the features corresponding to the correct exemplars for the current
			# parameter setting
			cur_exemplar_index = exemplar_index[(num_e, num_t)]
			cols = calculate_feature_col_indices(cur_exemplar_index, exemplar_offsets)
			feature_mat_cols = all_feature_mat[:,cols]
			for subset_size in subset_sizes:
				
				# select the correct subset of the data for initial clustering
				feature_mat = feature_mat_cols[:subset_size,:]
				subset = docs[:subset_size]

				distances = ['rf']
				if args.euclidean and feature_type == 'all':
					distances.append('euclidean')

				for distance in distances:
					# precompute similarity matrix for subset.  To be used for every K
					if distance == 'rf':
						sim_mat = calc_sim_matrix(feature_mat, distance)
					else:
						sim_mat = feature_mat

					for k in Ks:
						tag = "%s_%s %d %d %d %d" % (feature_type, distance, k, subset_size, num_e, num_t) 

						initial_clusters = initial_cluster(sim_mat, k, subset, distance)
						print_clusters(initial_clusters, title="Initial", tag="%s_%s" % ('init', tag))

						sclusters = cluster_refinement(initial_clusters, args)
						print_clusters(sclusters, title="Refine", tag="%s_%s" % ('refine', tag))

						bootstrap_clusters = bootstrap_cluster(sclusters, docs, subset_size)
						print_clusters(bootstrap_clusters, title="Bootstrap", tag="%s_%s" % ('bootstrap', tag))
						if args.no_refine:
							no_refine_bootstrap_clusters = bootstrap_cluster(initial_clusters, docs, subset_size)
							print_clusters(no_refine_bootstrap_clusters, title="No Refine", tag="%s_%s" % ('norefine', tag))
							


