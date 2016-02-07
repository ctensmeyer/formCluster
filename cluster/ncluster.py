
import os
import datetime
import metric
import cluster
import network2
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
import traceback
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
		min_size = int(min(args.minpts, args.minpts_perc * len(_cluster.members)))
	reachabilities = selector.OPTICS(dist_mat, min_size)
	indices = selector.separateClusters(reachabilities, min_size)

	# comes back as selector.dataPoint classes
	indices = map(lambda l: map(lambda dp: dp._id, l), indices)
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
	num_features = calc_num_features(seeds, feature_type)

	feature_mat = np.zeros( (num_docs, num_features) )
	offsets = list()
	offsets.append(0)
	for x, _doc in enumerate(docs):
		if x % 20 == 0 and _print:
			pass
			print "\t%d/%d (%.2f%%) Documents Extracted" % (x, num_docs, 100. * x / num_docs)
		offset = 0
		for seed in seeds:
			vector = _extract_features(_doc, seed, feature_type)
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


def print_summary(clusters, label, k, size_subset, num_seeds, mpts):
	acc, v = get_acc_v_measure(clusters)
	num_clusters = len(clusters)

	print "%s\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f" % (label, k, num_clusters, size_subset, num_seeds, mpts, acc, v)

def print_summary3(clusters, label, k, size_subset, num_seeds, args):
	acc, v = get_acc_v_measure(clusters)
	num_clusters = len(clusters)

	print "%s\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f" % (label, k, num_clusters, size_subset, num_seeds, mpts, acc, v)

def print_summary2(clusters, label, k, size_subset, num_seeds, mpts, num_types):
	acc, v = get_acc_v_measure(clusters)
	num_clusters = len(clusters)

	print "%s\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%d" % (label, k, num_clusters, size_subset, num_seeds, mpts, acc, v, num_types)
	
					

def run(docs, Ks, subsets, seeds, min_pts, init_only=False):
	'''
	Runs all specified combinations of parameters and logs the output.
	This is the serial version.
	'''
	random.shuffle(docs)
	largest_subset_size = subsets[-1]
	largest_num_seeds = seeds[-1]

	# this ties together all of the experiments
	largest_subset = docs[:largest_subset_size]
	all_seeds = random.sample(largest_subset, largest_num_seeds)
	largest_feature_mat, offsets = extract_features(largest_subset, all_seeds)

	for size_subset in subsets:
		subset = docs[:size_subset]
		for num_seeds in seeds:
			try:
				end_col = offsets[num_seeds]
				feature_mat = largest_feature_mat[:size_subset,:end_col]

				random_matrix = compute_random_matrix(feature_mat)
				rf = train_random_forest(feature_mat, random_matrix)
				sim_matrix = compute_sim_mat(feature_mat, rf)

				for k in Ks:
					try:
						assignments = spectral_cluster(sim_matrix, k)
						initial_clusters = form_clusters(subset, assignments)

						print "%s\n%s\n%s" % ("*" * 30, "Initial Clusters:", "*" * 30)
						print_cluster_analysis(initial_clusters)
						print_summary(initial_clusters, "init", k, size_subset, num_seeds, 0)

						if init_only:
							continue

						set_cluster_centers(initial_clusters)
						dist_mats = cluster_dist_mats(initial_clusters)
						for mpts in min_pts:
							try:
								sclusters = utils.flatten(map(lambda _cluster, dist_mat: split_cluster(_cluster, dist_mat, mpts), initial_clusters, dist_mats))
								print "%s\n%s: %d\n%s" % ("*" * 30, "Split Clusters", mpts, "*" * 30)
								print_cluster_analysis(sclusters)
								set_cluster_centers(sclusters)
								
								# get the features for final classification
								prototypes = map(lambda _cluster: _cluster.center, sclusters)
								features = extract_features(docs, prototypes)[0]
								training_labels = np.zeros(size_subset, dtype=np.int16)
								for x, _doc in enumerate(subset):
									for y,  _cluster in enumerate(sclusters):
										if _doc in _cluster.members:
											training_labels[x] = y
											break
								training_features = features[:size_subset,:]

								# final rf clustering
								rf = sklearn.ensemble.RandomForestClassifier(n_estimators=NUM_TREES, bootstrap=False, n_jobs=THREADS)
								rf.fit(training_features, training_labels)
								assignments = rf.predict(features)
								final_rf_clusters = form_clusters(docs, assignments)

								#final logistic
								lr = sklearn.linear_model.LogisticRegression(penalty='l1')
								lr.fit(training_features, training_labels)
								assignments = lr.predict(features)
								final_lr_clusters = form_clusters(docs, assignments)

								print "%s\n%s: %d\n%s" % ("*" * 30, "Final RF Clusters", mpts, "*" * 30)
								print_cluster_analysis(final_rf_clusters)
								print "%s\n%s: %d\n%s" % ("*" * 30, "Final LR Clusters", mpts, "*" * 30)
								print_cluster_analysis(final_lr_clusters)
								print_summary(sclusters, "split", k, size_subset, num_seeds, mpts)
								print_summary(final_rf_clusters, "fin_rf", k, size_subset, num_seeds, mpts)
								print_summary(final_lr_clusters, "fin_lr", k, size_subset, num_seeds, mpts)
							except:
								print "Error occured min_pts", (k, size_subset, num_seeds, mpts)
								print traceback.print_exc()	
					except:
						print "Error occured k", (k, size_subset, num_seeds, '?')
						print traceback.print_exc()	
			except:
				print "Error occured num_seeds", ('?', size_subset, num_seeds, '?')
				print traceback.print_exc()	

					
def run_no_split(docs, Ks, subsets, seeds):
	'''
	Runs all specified combinations of parameters and logs the output.
	This is the serial version that performs no splitting.
	'''
	random.shuffle(docs)
	largest_subset_size = subsets[-1]
	largest_num_seeds = seeds[-1]

	# this ties together all of the experiments
	largest_subset = docs[:largest_subset_size]
	all_seeds = random.sample(largest_subset, largest_num_seeds)
	largest_feature_mat, offsets = extract_features(largest_subset, all_seeds)

	for size_subset in subsets:
		subset = docs[:size_subset]
		for num_seeds in seeds:
			try:
				end_col = offsets[num_seeds]
				feature_mat = largest_feature_mat[:size_subset,:end_col]

				random_matrix = compute_random_matrix(feature_mat)
				rf = train_random_forest(feature_mat, random_matrix)
				sim_matrix = compute_sim_mat(feature_mat, rf)

				for k in Ks:
					try:
						assignments = spectral_cluster(sim_matrix, k)
						initial_clusters = form_clusters(subset, assignments)

						print "%s\n%s\n%s" % ("*" * 30, "Initial Clusters:", "*" * 30)
						print_cluster_analysis(initial_clusters)
						print_summary(initial_clusters, "init_no_split", k, size_subset, num_seeds, -1)

						set_cluster_centers(initial_clusters)
								
						# get the features for final classification
						prototypes = map(lambda _cluster: _cluster.center, initial_clusters)
						features = extract_features(docs, prototypes)[0]
						training_labels = np.zeros(size_subset, dtype=np.int16)
						for x, _doc in enumerate(subset):
							for y,  _cluster in enumerate(initial_clusters):
								if _doc in _cluster.members:
									training_labels[x] = y
									break
						training_features = features[:size_subset,:]

						# final rf clustering
						rf = sklearn.ensemble.RandomForestClassifier(n_estimators=NUM_TREES, bootstrap=False, n_jobs=THREADS)
						rf.fit(training_features, training_labels)
						assignments = rf.predict(features)
						final_rf_clusters = form_clusters(docs, assignments)

						# final logistic
						lr = sklearn.linear_model.LogisticRegression(penalty='l1')
						lr.fit(training_features, training_labels)
						assignments = lr.predict(features)
						final_lr_clusters = form_clusters(docs, assignments)

						print "%s\n%s: %d\n%s" % ("*" * 30, "Final RF Clusters", -1, "*" * 30)
						print_cluster_analysis(final_rf_clusters)
						print "%s\n%s: %d\n%s" % ("*" * 30, "Final LR Clusters", -1, "*" * 30)
						print_cluster_analysis(final_lr_clusters)
						print_summary(final_rf_clusters, "fin_rf_no_split", k, size_subset, num_seeds, -1)
						print_summary(final_lr_clusters, "fin_lr_no_split", k, size_subset, num_seeds, -1)
					except:
						print "Error occured k", (k, size_subset, num_seeds)
						print traceback.print_exc()	
			except:
				print "Error occured num_seeds", ('?', size_subset, num_seeds)
				print traceback.print_exc()	

					
def run_auto_minpts(docs, Ks, subsets, seeds):
	'''
	Runs all specified combinations of parameters and logs the output.
	This is the serial version.
	'''
	random.shuffle(docs)
	largest_subset_size = subsets[-1]
	largest_num_seeds = seeds[-1]

	# this ties together all of the experiments
	largest_subset = docs[:largest_subset_size]
	all_seeds = random.sample(largest_subset, largest_num_seeds)
	largest_feature_mat, offsets = extract_features(largest_subset, all_seeds)

	for size_subset in subsets:
		subset = docs[:size_subset]
		for num_seeds in seeds:
			try:
				end_col = offsets[num_seeds]
				feature_mat = largest_feature_mat[:size_subset,:end_col]

				random_matrix = compute_random_matrix(feature_mat)
				rf = train_random_forest(feature_mat, random_matrix)
				sim_matrix = compute_sim_mat(feature_mat, rf)

				for k in Ks:
					try:
						assignments = spectral_cluster(sim_matrix, k)
						initial_clusters = form_clusters(subset, assignments)

						print "%s\n%s\n%s" % ("*" * 30, "Initial Clusters:", "*" * 30)
						print_cluster_analysis(initial_clusters)
						print_summary(initial_clusters, "init", k, size_subset, num_seeds, -2)

						set_cluster_centers(initial_clusters)
						dist_mats = cluster_dist_mats(initial_clusters)
						sclusters = utils.flatten(map(lambda _cluster, dist_mat: split_cluster_auto(_cluster, dist_mat), initial_clusters, dist_mats))
						print "%s\n%s: %d\n%s" % ("*" * 30, "Split Clusters", -2, "*" * 30)
						print_cluster_analysis(sclusters)
						set_cluster_centers(sclusters)
						
						# get the features for final classification
						prototypes = map(lambda _cluster: _cluster.center, sclusters)
						features = extract_features(docs, prototypes)[0]
						training_labels = np.zeros(size_subset, dtype=np.int16)
						for x, _doc in enumerate(subset):
							for y,  _cluster in enumerate(sclusters):
								if _doc in _cluster.members:
									training_labels[x] = y
									break
						training_features = features[:size_subset,:]

						# final rf clustering
						rf = sklearn.ensemble.RandomForestClassifier(n_estimators=NUM_TREES, bootstrap=False, n_jobs=THREADS)
						rf.fit(training_features, training_labels)
						assignments = rf.predict(features)
						final_rf_clusters = form_clusters(docs, assignments)

						#final logistic
						lr = sklearn.linear_model.LogisticRegression(penalty='l1')
						lr.fit(training_features, training_labels)
						assignments = lr.predict(features)
						final_lr_clusters = form_clusters(docs, assignments)

						print "%s\n%s: %d\n%s" % ("*" * 30, "Final RF Clusters", -2, "*" * 30)
						print_cluster_analysis(final_rf_clusters)
						print "%s\n%s: %d\n%s" % ("*" * 30, "Final LR Clusters", -2, "*" * 30)
						print_cluster_analysis(final_lr_clusters)
						print_summary(sclusters, "split", k, size_subset, num_seeds, -2)
						print_summary(final_rf_clusters, "fin_rf", k, size_subset, num_seeds, -2)
						print_summary(final_lr_clusters, "fin_lr", k, size_subset, num_seeds, -2)
					except:
						print "Error occured k", (k, size_subset, num_seeds)
						print traceback.print_exc()	
			except:
				print "Error occured num_seeds", ('?', size_subset, num_seeds)
				print traceback.print_exc()	

def run_type(docs, Ks, subsets, seeds, types):
	print types
	random.shuffle(docs)

	for size_subset in subsets:
		subset = docs[:size_subset]
		sorted_docs = list()
		for _doc in docs:
			for sublist in sorted_docs:
				if sublist[0].label == _doc.label:
					sublist.append(_doc)
					break
			else:
				sorted_docs.append([_doc])
		sorted_docs.sort(key=len, reverse=True)
		for sublist in sorted_docs:
			random.shuffle(sublist)

		print "Num sublists:", len(sorted_docs)
		for num_seeds in seeds:
			for num_types in types:
				try:
					# choose the seeds
					lseeds = list()
					idx = 0
					while len(lseeds) < num_seeds:
						type_idx = idx % num_types
						sublist = sorted_docs[type_idx]
						sublist_idx = idx / num_types
						if len(sublist) > sublist_idx:
							lseeds.append(sublist[sublist_idx])
						idx += 1

					feature_mat = extract_features(subset, lseeds)[0]
					random_matrix = compute_random_matrix(feature_mat)
					rf = train_random_forest(feature_mat, random_matrix)
					sim_matrix = compute_sim_mat(feature_mat, rf)

					for k in Ks:
						try:
							assignments = spectral_cluster(sim_matrix, k)
							initial_clusters = form_clusters(subset, assignments)

							print "%s\n%s\n%s" % ("*" * 30, "Initial Clusters:", "*" * 30)
							print_cluster_analysis(initial_clusters)
							print_summary2(initial_clusters, "init_type", k, size_subset, num_seeds, -2, num_types)

							set_cluster_centers(initial_clusters)
							dist_mats = cluster_dist_mats(initial_clusters)
							sclusters = utils.flatten(map(lambda _cluster, dist_mat: split_cluster_auto(_cluster, dist_mat), initial_clusters, dist_mats))
							print "%s\n%s: %d\n%s" % ("*" * 30, "Split Clusters", -2, "*" * 30)
							print_cluster_analysis(sclusters)
							set_cluster_centers(sclusters)
							
							# get the features for final classification
							prototypes = map(lambda _cluster: _cluster.center, sclusters)
							features = extract_features(docs, prototypes)[0]
							training_labels = np.zeros(size_subset, dtype=np.int16)
							for x, _doc in enumerate(subset):
								for y,  _cluster in enumerate(sclusters):
									if _doc in _cluster.members:
										training_labels[x] = y
										break
							training_features = features[:size_subset,:]

							# final rf clustering
							rf = sklearn.ensemble.RandomForestClassifier(n_estimators=NUM_TREES, bootstrap=False, n_jobs=THREADS)
							rf.fit(training_features, training_labels)
							assignments = rf.predict(features)
							final_rf_clusters = form_clusters(docs, assignments)

							#final logistic
							lr = sklearn.linear_model.LogisticRegression(penalty='l1')
							lr.fit(training_features, training_labels)
							assignments = lr.predict(features)
							final_lr_clusters = form_clusters(docs, assignments)

							print "%s\n%s: %d\n%s" % ("*" * 30, "Final RF Clusters", -2, "*" * 30)
							print_cluster_analysis(final_rf_clusters)
							print "%s\n%s: %d\n%s" % ("*" * 30, "Final LR Clusters", -2, "*" * 30)
							print_cluster_analysis(final_lr_clusters)
							print_summary2(sclusters, "split_type", k, size_subset, num_seeds, -2, num_types)
							print_summary2(final_rf_clusters, "fin_rf_type", k, size_subset, num_seeds, -2, num_types)
							print_summary2(final_lr_clusters, "fin_lr_type", k, size_subset, num_seeds, -2, num_types)
						except:
							print "Error occured k", (k, size_subset, num_seeds)
							print traceback.print_exc()	
				except:
					print "Error occured num_seeds", ('?', size_subset, num_seeds)
					print traceback.print_exc()	

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
			for type_idx in num_t:
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

	return all_examplars, exemplar_index
	
	

def get_exemplars(docs, num_exemplars, num_types):
	if not num_types:
		return get_random_exemplars(docs, num_exemplars)
	else:
		return get_oracle_exemplars(docs, num_exemplars, num_types)

def calculate_feature_col_indices(index, offsets):
	cols = list()
	for idx in index:
		start = offsets[idx]
		stop = offsets[idx+1]
		cols += range(start,stop)
	cols.sort()
	return cols

def cluster_refinement(initial_clustering):
	set_cluster_centers(initial_clusters)
	dist_mats = cluster_dist_mats(initial_clusters)
	sclusters = utils.flatten(map(
		lambda _cluster, dist_mat: split_cluster_auto(_cluster, dist_mat), 
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
	training_features = bootstrap_features[:size_subset,:]
	
	return bootstrap_features, training_features, training_labels 

def rf_sim_mat(feature_mat):
	random_matrix = compute_random_matrix(feature_mat)
	rf = train_random_forest(feature_mat, random_matrix)
	sim_matrix = compute_sim_mat(feature_mat, rf)
	return sim_matrix

def euclidean_sim_mat(feature_mat):
	dists = scipy.spatial.distance.pdist(features, 'euclidean')
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
	assignments = lr.predict(features)
	bootstrap_clusters = form_clusters(docs, assignments)

	return bootstrap_clusters 

def initial_cluster(sim_matrix, k, subset):
	assignments = spectral_cluster(sim_matrix, k)
	initial_clusters = form_clusters(subset, assignments)
	return initial_clusters 

def print_clusters(clusters, title, tag):
	print "%s\n%s\n%s" % ("*" * 30, "%s Clusters:" % title, "*" * 30)
	print_cluster_analysis(clusters)

	acc, v = get_acc_v_measure(clusters)
	num_clusters = len(clusters)
	print "%s %.5f %.5f %d" % (tag, acc, v, num_clusters)
	

def confirm(docs, Ks, subset_sizes, num_exemplars, num_types, args):
	largest_subset_size = max(subset_sizes)
	largest_subset = docs[:largest_subset_size]

	# all_exemplars is a list of unique exemplars used across all specified parameter settings
	# of num_exemplars and num_types.  This allows feature extraction to be done once to save
	# redundant computation between experiments.
	# exemplar_index is a dictionary whose entry (num_e, num_t) is a list of indices into
	# all_exemplars.  When random exemplars are used, $num_t = 0 as a sentinel value
	all_exemplars, exemplar_index = get_exemplars(largest_subset, num_exemplars, num_types)

	# precompute features for all docs in a subset with all exemplars used
	feature_types = ['all']
	if args.text_only:
		feature_types.append('text')
	if args.rule_only:
		feature_types.append('rule')

	for feature_type in feature_types:
		all_feature_mat, exemplar_offsets = extract_features(largest_subset, all_exemplars, feature_type)

		for num_e, num_t in exemplar_index.keys():
			
			# select the features corresponding to the correct exemplars for the current
			# parameter setting
			cur_exemplar_index = exemplar_index[(num_e, num_t)]
			cols = calculate_feature_col_indices(cur_exemplar_index, exemplar_offsets)
			feature_mat_cols = all_feature_mat[:,cols]
			for subset_size in subset_sizes:
				
				# select the correct subset of the data for initial clustering
				feature_mat = feature_mat_cols[:subset_size,:]

				distances = ['rf']
				if args.euclidean:
					distances.append('euclidean')

				for distance in distances:
					# precompute similarity matrix for subset.  To be used for every K
					sim_matrix = calc_sim_matrix(feature_mat, distance)

					for K in Ks:
						tag = "%s_%s %d %d %d %d" % (feature_type, distance, k, subset_size, num_e, num_t) 

						initial_clusters = initial_cluster(sim_matrix, k, subset)
						print_clusters(initial_clusters, title="Initial", tag="%s_%s" % ('init', tag))

						sclusters = cluster_refinement(initial_clustering)
						print_clusters(sclusters, title="Refine", tag="%s_%s" % ('refine', tag))

						bootstrap_clusters = bootstrap_cluster(sclusters, docs, subset_size)
						print_clusters(bootstrap_clusters, title="Bootstrap", tag="%s_%s" % ('bootstrap', tag))
						if args.no_refine:
							no_refine_bootstrap_clusters = bootstrap_cluster(initial_clusters, docs, subset_size)
							print_clusters(no_refine_bootstrap_clusters, title="No Refine", tag="%s_%s" % ('norefine', tag))
							


