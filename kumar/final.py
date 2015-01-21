
import os
import sys
import cv2
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

_image_ext = ".jpg"

# SURF Extraction
_surf_upright = True
_surf_extended = False
_surf_threshold = 10000
_num_surf_features = 10000

# Codebook & Features
_codebook_sizes = [100, 200]
_num_trials = 2
_perc_docs_for_codebook = 0.05
_num_surf_features_codebook = 30000
_max_k_medoids_iters = 30
_H_partitions = 4
_V_partitions = 3

# Random Forest
_num_trees = 2000
_rf_threads = 7
_perc_random_data = 1

# Spectral Clustering
_cluster_range = (2, int(sys.argv[3]))
_assignment_method = 'discretize'


# not parameters
_num_histograms = ( (2 ** (_H_partitions) - 1) + (2 ** (_V_partitions) - 1) - 1 )
_surf_instance = cv2.SURF(_surf_threshold)
_surf_instance.upright = _surf_upright
_surf_instance.extended = _surf_extended
_surf_features = dict()
_print_interval = 20
_output_file = sys.argv[2]
_recorded_metrics = ['Codebook_Size', 'Trial_Num', 'Num_Clusters', 'Num_Trees', 
					'Acc', 'V-measure', 'Completeness', 'Homogeneity', 'ARI', 'Silhouette']

_out = open(_output_file, 'w')
_out.write("%s\n" % "\t".join(_recorded_metrics))


def calc_surf_features(im_file):
	if im_file not in _surf_features:
		im = cv2.imread(im_file, 0)
		height = im.shape[0]
		width = im.shape[1]
		# surf_features[0] is the array of keypoints
		# surf_features[1] is the array of descriptors
		_surf_instance.hessianThreshold = _surf_threshold
		kps, deses = _surf_instance.detectAndCompute(im, None)
		while len(kps) < _num_surf_features:
			_surf_instance.hessianThreshold /= 2
			kps, deses = _surf_instance.detectAndCompute(im, None)
		pts = np.array(map(lambda kp: kp.pt, kps[:_num_surf_features]))
		deses = deses[:_num_surf_features]
		_surf_features[im_file] = (pts, deses, width, height)
	return _surf_features[im_file]
	
def calc_features(im_file, codebook):
	# calc the surf features
	codebook_size = len(codebook)
	num_features = codebook_size * _num_histograms
	pts, deses, width, height = calc_surf_features(im_file)

	# the most fine grained partitions
	horz_histos = np.zeros(2 ** (_H_partitions - 1) * codebook_size)
	vert_histos = np.zeros(2 ** (_V_partitions - 1) * codebook_size)
	horz_stride = (width / 2 ** (_H_partitions - 1)) + 1
	vert_stride = (height / 2 ** (_V_partitions - 1)) + 1

	#print "Image dims", (width, height)
	#print "horz_stride", horz_stride
	#print "vert_stride", vert_stride
	#print "horz_histo shape", horz_histos.shape
	#print "vert_histo shape", vert_histos.shape

	closest_code = lambda feature: scipy.spatial.distance.cdist(codebook, [feature], metric='cityblock').argmin()

	# populate the most fine grained partitions
	for pt, des in zip(pts, deses):
		#print pt
		#print des
		idx = closest_code(des)
		#print "Closest code:", idx
		horz_histos[idx + ( int(pt[0] / horz_stride)  * codebook_size)] += 1
		vert_histos[idx + ( int(pt[1] / vert_stride)  * codebook_size)] += 1

	#print "Horz Histo:"
	#print horz_histos
	#print
	#print "Vert Histo:"
	#print vert_histos

	# aggregate the statistics
	histos = np.zeros(num_features)
	histo_offset = 0
	stride = 1
	while stride <= horz_histos.shape[0] / codebook_size:
		horz_offset = 0
		while horz_offset < horz_histos.shape[0] / codebook_size:
			horz_cur = horz_offset
			while horz_cur < (horz_offset + stride):
				histos[histo_offset * codebook_size : (histo_offset + 1) * codebook_size] += (
					horz_histos[horz_cur * codebook_size: (horz_cur + 1) * codebook_size])
				horz_cur += 1
			horz_offset += stride
			histo_offset += 1
		stride *= 2
	stride = 1
	while stride < vert_histos.shape[0] / codebook_size:
		vert_offset = 0
		while vert_offset < vert_histos.shape[0] / codebook_size:
			vert_cur = vert_offset
			while vert_cur < (vert_offset + stride):
				histos[histo_offset * codebook_size : (histo_offset + 1) * codebook_size] += (
					vert_histos[vert_cur * codebook_size: (vert_cur + 1) * codebook_size])
				vert_cur += 1
			vert_offset += stride
			histo_offset += 1
		stride *= 2

	#print
	#print "Whole Histo"
	#print histos

	# normalize
	for offset in xrange(num_features / codebook_size):
		s = histos[offset * codebook_size : (offset + 1) * codebook_size].sum()
		if s:
			histos[offset * codebook_size : (offset + 1) * codebook_size] /= float(s)
	features = histos

	#print
	#print "Normalized"
	#print self.features
	#exit()
	return features

# an instance is a (im_filename, label) tuple
def load_instances(in_dir):
	print "Loading Instances"

	instances = list()
	label_map = dict()
	next_id = 0
	for sdir in os.listdir(in_dir):
		rdir = os.path.join(in_dir, sdir)
		for im_file in os.listdir(rdir):
			if im_file.endswith(_image_ext):
				_label = label.preprocess_label(sdir)
				if not _label in label_map:
					label_map[_label] = next_id
					next_id += 1
				label_id = label_map[_label]
				instances.append( (os.path.join(rdir, im_file), label_id) )
	random.shuffle(instances)

	return instances

def sample_surf_features(instances):
	print "Getting Surf Features for Codebook Construction"
	num_to_sample = int(len(instances) * _perc_docs_for_codebook)
	sampled_instances = random.sample(instances, num_to_sample)
	print "\tSampling Features from %d instances" % num_to_sample
	list_sampled_features = list()
	for instance in sampled_instances:
		deses = calc_surf_features(instance[0])[1]
		list_sampled_features.append(deses)
	sampled_features = np.concatenate(list_sampled_features)
	np.random.shuffle(sampled_features)
	return sampled_features[:_num_surf_features_codebook]

def construct_codebook(instances, codebook_size):
	print "Constructing Codebook"

	surf_feature_samples = sample_surf_features(instances)

	distances = scipy.spatial.distance.pdist(surf_feature_samples, 'cityblock')
	distances = scipy.spatial.distance.squareform(distances)

	indices = kmedoids.cluster(distances, k=codebook_size, maxIters=_max_k_medoids_iters)[1]
	codebook = surf_feature_samples[indices]

	print "Done\n"
	return codebook

def compute_features(instances, codebook):
	print "Computing histogram features for each instance"

	features = list()
	total = len(instances)
	for instance in instances:
		features.append(calc_features(instance[0], codebook))
	features = np.array(features)

	print "Done\n"
	return features


def compute_random_matrix(data_matrix):
	print "Constructing Random Training Set"

	rand_shape = (int(data_matrix.shape[0] * _perc_random_data), data_matrix.shape[1])
	rand_mat = np.zeros(rand_shape)
	#np.random.seed(12345)
	for col in xrange(rand_mat.shape[1]):
		vals = data_matrix[:,col]
		for row in xrange(rand_mat.shape[0]):
			rand_mat[row, col] = np.random.choice(vals)

	print "Done\n"
	return rand_mat

def train_classifier(real_data, fake_data):
	print "Training Random Forest"

	rf = sklearn.ensemble.RandomForestClassifier(n_estimators=_num_trees, max_features='auto',
												bootstrap=False, n_jobs=_rf_threads)
	combined_data = np.concatenate( (real_data, fake_data) )
	labels = np.concatenate( (np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])) )
	rf.fit(combined_data, labels)

	print "Done\n"
	return rf

def compute_sim_mat(data_matrix, random_forest):
	print "Computing the Similarity Matrix"

	leaf_nodes = random_forest.apply(data_matrix)
	sim_mat = scipy.spatial.distance.pdist(leaf_nodes, "hamming")
	sim_mat = scipy.spatial.distance.squareform(sim_mat)
	sim_mat = 1 - sim_mat

	print "Done\n"
	return sim_mat

def spectral_cluster(affinity_matrix, num_clusters):
	print "Performing Spectral Clustering"

	sc = sklearn.cluster.SpectralClustering(n_clusters=num_clusters, affinity="precomputed",
											assign_labels=_assignment_method)
	assignments = sc.fit_predict(affinity_matrix)

	print "Done\n"
	return assignments

def calc_acc(true_labels, predicted_labels):
	_num_clusters = predicted_labels.max() + 1
	counters = {x: collections.Counter() for x in xrange(_num_clusters)}
	for true_label, predicted_label in zip(true_labels, predicted_labels):
		counters[predicted_label][true_label] += 1

	num_correct = 0
	for counter in counters.values():
		num_correct += counter.most_common(1)[0][1]

	return num_correct / float(len(true_labels))

def main(in_dir):
	instances = load_instances(in_dir)
	true_labels = map(lambda tup: tup[1], instances)
	print "Num Instanes", len(instances)

	for x, instance in enumerate(instances):
		calc_surf_features(instance[0])
		if x % _print_interval == 0:
			print "\t%d/%d (%2.1f%%) Documents processed" % (x, len(instances), 100.0 * x / len(instances))

	total_trials = len(_codebook_sizes) * _num_trials
	for y, codebook_size in enumerate(_codebook_sizes):
		for trial in xrange(_num_trials):
			trial_num = trial + y * len(_codebook_sizes)
			print "\t%d/%d (%2.1f%%) Trials processed" % (trial_num, total_trials, 100.0 * trial_num / total_trials)

			codebook = construct_codebook(instances, codebook_size)
			data_matrix = compute_features(instances, codebook)
			random_matrix = compute_random_matrix(data_matrix)
			random_forest = train_classifier(data_matrix, random_matrix)
			sim_mat = compute_sim_mat(data_matrix, random_forest)

			for num_clusters in xrange(_cluster_range[0], _cluster_range[1] + 1):
				predicted_labels = spectral_cluster(sim_mat, num_clusters)
				acc = calc_acc(true_labels, predicted_labels)
				h, c, v = sklearn.metrics.homogeneity_completeness_v_measure(true_labels, predicted_labels)
				ari = sklearn.metrics.adjusted_rand_score(true_labels, predicted_labels)
				silhouette = sklearn.metrics.silhouette_score(sim_mat, predicted_labels, metric='precomputed')
				_out.write("%s\n" % "\t".join(
					map(lambda x: "%d" % x, [codebook_size, trial, num_clusters, _num_trees]) +
					map(lambda x: "%.3f" % x, [acc, v, c, h, ari, silhouette])))
	_out.close()


if __name__ == "__main__":
	in_dir = sys.argv[1]
	main(in_dir)

