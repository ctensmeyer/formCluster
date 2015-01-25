
import os
import sys
import math
import label
import random
import shutil
import metric
import cluster
import numpy as np
import collections
import sklearn.cluster
import sklearn.ensemble
import scipy.spatial.distance

np.set_printoptions(precision=2, linewidth=200, suppress=True)

# this must be set carefully so the size of the codebook comes out right
_num_histograms = 21
_num_trials = 5

# Random Forest
_num_trees = 1000
_rf_threads = 5
_percs_random_data = [.1, .25, .5, 1., 1.5, 2, 3]
#_percs_random_data = [.1, 1., 2]

# not parameters
_output_file = sys.argv[2]
_recorded_metrics = ['Matrix_File', 'Codebook_Size', 'Trial_Num', 'Num_Trees', 'Perc_Rand', 'Silhouette']

_out = open(_output_file, 'w')
_out.write("%s\n" % "\t".join(_recorded_metrics))
_verbose = True


def compute_random_matrix(data_matrix, rand_perc):
	rand_shape = (int(data_matrix.shape[0] * rand_perc), data_matrix.shape[1])
	rand_mat = np.zeros(rand_shape)
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

def compute_dissim_mat(data_matrix, random_forest):

	leaf_nodes = random_forest.apply(data_matrix)
	dissim_mat = scipy.spatial.distance.pdist(leaf_nodes, "hamming")
	dissim_mat = scipy.spatial.distance.squareform(dissim_mat)

	return dissim_mat

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
	clusters = cluster_map.values()
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
		if f.startswith("data_matrix") and f.endswith(".npy"):
			data_matrix_files.append(os.path.join(in_dir, f))
	data_matrix_files.sort()
	num_matrices = len(data_matrix_files)
	work_items = _num_trials * len(_percs_random_data)

	for y, data_matrix_file in enumerate(data_matrix_files):
		print "\t%d/%d (%2.1f%%) Matrices processed" % (y, num_matrices, 100.0 * y / num_matrices)
		
		f = open(data_matrix_file)
		data_matrix = np.load(f)
		f.close()
		codebook_size = data_matrix.shape[1] / _num_histograms
		work_items_completed = 0
		for trial_num in xrange(_num_trials):
			for rand_perc in _percs_random_data:
				print "\t\t%d/%d (%2.1f%%) Work Items processed" % (work_items_completed, work_items, 
																100.0 * work_items_completed / work_items)
				random_matrix = compute_random_matrix(data_matrix, rand_perc)
				random_forest = train_classifier(data_matrix, random_matrix)
				dissim_mat = compute_dissim_mat(data_matrix, random_forest)
				silhouette = sklearn.metrics.silhouette_score(dissim_mat, true_labels, metric='precomputed')
				f = os.path.basename(data_matrix_file)
				s = "%s\t%d\t%d\t%d\t%.3f\t%.3f\n" % (f, codebook_size, trial_num, _num_trees, rand_perc, silhouette)
				_out.write(s)
				print s
				work_items_completed += 1
	_out.close()


if __name__ == "__main__":
	in_dir = sys.argv[1]
	main(in_dir)

