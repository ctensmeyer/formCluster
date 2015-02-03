
import os
import sys
import math
import label
import random
import shutil
import metric
import cPickle
import cluster
import kmedoids
import numpy as np
import collections
import sklearn.cluster
import sklearn.ensemble
import scipy.spatial.distance

np.set_printoptions(precision=2, linewidth=200, suppress=True)

# this must be set carefully so the size of the codebook comes out right
_num_histograms = 21

# Random Forest
_num_trees = 2000
_rf_threads = 5
_perc_random_data = 1

# Spectral Clustering
_cluster_range = (2, int(sys.argv[3]))
_assignment_method = 'discretize'


# not parameters
_output_file = sys.argv[2]
_recorded_metrics = ['Matrix_File', 'Codebook_Size', 'Trial_Num', 'Num_Clusters', 'Num_Trees', 
					'Acc', 'V-measure', 'Completeness', 'Homogeneity', 'ARI', 'Obs_Silh', 'True_Silh']

_out = open(_output_file, 'w')
_out.write("%s\n" % "\t".join(_recorded_metrics))
_verbose = True
#_prefix = "rand"
_prefix = "data_matrix"


def compute_random_matrix(data_matrix):
	rand_shape = (int(data_matrix.shape[0] * _perc_random_data), data_matrix.shape[1])
	rand_mat = np.zeros(rand_shape)
	#np.random.seed(12345)
	for col in xrange(rand_mat.shape[1]):
		vals = data_matrix[:,col]
		for row in xrange(rand_mat.shape[0]):
			rand_mat[row, col] = np.random.choice(vals)

	return rand_mat

def train_classifier(real_data, fake_data):
	rf = sklearn.ensemble.RandomForestClassifier(n_estimators=_num_trees, max_features='auto',
												bootstrap=False, n_jobs=_rf_threads)
	combined_data = np.concatenate( (real_data, fake_data) )
	labels = np.concatenate( (np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])) )
	rf.fit(combined_data, labels)

	return rf

def compute_sim_mat(data_matrix, random_forest):

	leaf_nodes = random_forest.apply(data_matrix)
	sim_mat = scipy.spatial.distance.pdist(leaf_nodes, "hamming")
	sim_mat = scipy.spatial.distance.squareform(sim_mat)
	sim_mat = 1 - sim_mat

	return sim_mat

def spectral_cluster(affinity_matrix, num_clusters):

	sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters, affinity="precomputed",
											assign_labels=_assignment_method)
	assignments = sc.fit_predict(affinity_matrix)

	return assignments

def calc_acc(true_labels, predicted_labels):
	_num_clusters = predicted_labels.max() + 1
	counters = {x: collections.Counter() for x in xrange(_num_clusters)}
	for true_label, predicted_label in zip(true_labels, predicted_labels):
		counters[predicted_label][true_label] += 1

	num_correct = 0
	for counter in counters.values():
		if counter:
			num_correct += counter.most_common(1)[0][1]

	return num_correct / float(len(true_labels))

def form_clusters(true_labels, predicted_labels):
	cluster_map = dict()
	class Mock:
		pass
	for x in xrange(predicted_labels.max() + 1):
		m = Mock()
		m.label = None
		cluster_map[x] = cluster.Cluster(list(), m, x)
	count = 0
	for true, predicted in zip(true_labels, predicted_labels):
		m = Mock()
		m._id = count
		count += 1
		m.label = true
		cluster_map[predicted].members.append(m)
	clusters = filter(lambda cluster: cluster.members, cluster_map.values())
	map(lambda cluster: cluster.set_label(), clusters)
	return clusters

def print_analysis(clusters):
	class Mock:
		pass
	m = Mock()
	m.get_clusters = lambda: clusters
	analyzer = metric.KnownClusterAnalyzer(m)
	analyzer.print_general_info()
	analyzer.print_histogram_info()
	analyzer.print_label_conf_mat()
	analyzer.print_label_cluster_mat()
	analyzer.print_label_info()
	analyzer.print_metric_info()

def main(in_dir):

	data_matrix_files = list()
	f = open(os.path.join(in_dir, "labels.npy"))
	true_labels = np.load(f)
	f.close()
	for f in os.listdir(in_dir):
		if f.startswith(_prefix) and f.endswith(".npy"):
			data_matrix_files.append(os.path.join(in_dir, f))
	data_matrix_files.sort()
	num_trials = len(data_matrix_files)

	for y, data_matrix_file in enumerate(data_matrix_files):
		print "\t%d/%d (%2.1f%%) Trials processed" % (y, num_trials, 100.0 * y / num_trials)
		
		f = open(data_matrix_file)
		data_matrix = np.load(f)
		f.close()
		codebook_size = data_matrix.shape[1] / _num_histograms
		random_matrix = compute_random_matrix(data_matrix)
		random_forest = train_classifier(data_matrix, random_matrix)
		sim_mat = compute_sim_mat(data_matrix, random_forest)

		for num_clusters in xrange(_cluster_range[0], _cluster_range[1] + 1):
			predicted_labels = spectral_cluster(sim_mat, num_clusters)
			acc = calc_acc(true_labels, predicted_labels)
			h, c, v = sklearn.metrics.homogeneity_completeness_v_measure(true_labels, predicted_labels)
			ari = sklearn.metrics.adjusted_rand_score(true_labels, predicted_labels)
			predicted_silhouette = sklearn.metrics.silhouette_score(1 - sim_mat, predicted_labels, metric='precomputed')
			actual_silhouette = sklearn.metrics.silhouette_score(1 - sim_mat, true_labels, metric='precomputed')
			f = os.path.basename(data_matrix_file)
			s = ("%s\n" % "\t".join([f] +
				map(lambda x: "%d" % x, [codebook_size, y, num_clusters, _num_trees]) +
				map(lambda x: "%.3f" % x, [acc, v, c, h, ari, predicted_silhouette, actual_silhouette])))
			_out.write(s)
			if _verbose:
				try:
					print s
					clusters = form_clusters(true_labels, predicted_labels)
					print_analysis(clusters)
				except Exception as e:
					print "Error occured in verbose output"
					print e
					print sys.exc_info()[0]

	_out.close()


if __name__ == "__main__":
	in_dir = sys.argv[1]
	main(in_dir)

