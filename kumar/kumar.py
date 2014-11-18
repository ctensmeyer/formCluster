
import os
import sys
import cv2
import math
import random
import shutil
import metric
import cPickle
import cluster
import kmedoids
import collections
import numpy as np
import scipy.spatial.distance
import sklearn.ensemble
import sklearn.cluster

np.set_printoptions(precision=2, linewidth=200, suppress=True)

_image_ext = ".jpg"

# Model Parameters
_surf_upright = True
_surf_extended = False
_surf_threshold = 10000
_surf_threshold_low = 3000
_num_surf_features = 10000
_min_surf_features = 5000

_codebook_size = 300
_perc_docs_for_codebook = 0.05
_max_surf_features = _codebook_size * 100
_H_partitions = 3
_V_partitions = 4
_use_k_medoids = True
_max_k_medoids_iters = 30

_num_features = _codebook_size * ( (2 ** (_H_partitions) - 1) + (2 ** (_V_partitions) - 1) - 1 )
print "Codebook Size", _codebook_size
print "Num Features", _num_features

_num_trees = 2000
_num_tree_features = int(math.sqrt(_num_features))


_surf_instance = cv2.SURF(_surf_threshold)
_surf_instance.upright = _surf_upright
_surf_instance.extended = _surf_extended

_surf_instance_low = cv2.SURF(_surf_threshold_low)
_surf_instance_low.upright = _surf_upright
_surf_instance_low.extended = _surf_extended

_number_of_clusters = 5

_print_interval = 20

# Impl Parameters
_clear_cache = False
_read_cache = True
_write_cache = True
_cache_dir = ".kumar_cache"
_cache_instances_file = os.path.join(_cache_dir, "instances.pckl")
_cache_codebook_file = os.path.join(_cache_dir, "codebook.pckl")
_cache_data_matrix_file = os.path.join(_cache_dir, "data_matrix.pckl")
_cache_random_matrix_file = os.path.join(_cache_dir, "random_matrix.pckl")
_cache_rf_file = os.path.join(_cache_dir, "rf.pckl")
_cache_sim_matrix_file = os.path.join(_cache_dir, "sim_matrix.pckl")
_cache_cluster_assignments_file = os.path.join(_cache_dir, "assignments.pckl")

if _clear_cache:
	try:
		shutil.rmtree(_cache_dir)
	except:
		print "Could not clear the cache"

try:
	os.makedirs(_cache_dir)
except:
	pass

class Instance:
	
	def __init__(self, im_file, label):
		self.im_file = im_file
		self.label = label
		self._id = os.path.splitext(im_file)[0]
		self.surf_features = None

	def calc_surf_features(self):
		if self.surf_features is None:
			im = cv2.imread(self.im_file, 0)
			self.height = im.shape[0]
			self.width = im.shape[1]
			# surf_features[0] is the array of keypoints
			# surf_features[1] is the array of descriptors
			kps, deses = _surf_instance.detectAndCompute(im, None)
			if len(kps) < _min_surf_features:
				kps, deses = _surf_instance_low.detectAndCompute(im, None)
			self.pts = np.array(map(lambda kp: kp.pt, kps[:_num_surf_features]))
			self.deses = deses[:_num_surf_features] + 0
			#print len(self.pts)
			#print self.deses.base is deses
			del deses

	def calc_features(self, codebook):
		# the most fine grained partitions
		horz_histos = np.zeros(2 ** (_H_partitions - 1) * _codebook_size)
		vert_histos = np.zeros(2 ** (_V_partitions - 1) * _codebook_size)
		self.calc_surf_features()
		horz_stride = (self.width / 2 ** (_H_partitions - 1)) + 1
		vert_stride = (self.height / 2 ** (_V_partitions - 1)) + 1

		#print "Image dims", (self.width, self.height)
		#print "horz_stride", horz_stride
		#print "vert_stride", vert_stride
		#print "horz_histo shape", horz_histos.shape
		#print "vert_histo shape", vert_histos.shape

		closest_code = lambda feature: scipy.spatial.distance.cdist(codebook, [feature], metric='cityblock').argmin()

		# populate the most fine grained partitions
		for pt, des in zip(self.pts, self.deses):
			#print pt
			#print des
			idx = closest_code(des)
			#print "Closest code:", idx
			horz_histos[idx + ( int(pt[0] / horz_stride)  * _codebook_size)] += 1
			vert_histos[idx + ( int(pt[1] / vert_stride)  * _codebook_size)] += 1

		#print "Horz Histo:"
		#print horz_histos
		#print
		#print "Vert Histo:"
		#print vert_histos

		# aggregate the statistics
		histos = np.zeros(_num_features)
		histo_offset = 0
		stride = 1
		while stride <= horz_histos.shape[0] / _codebook_size:
			horz_offset = 0
			while horz_offset < horz_histos.shape[0] / _codebook_size:
				horz_cur = horz_offset
				while horz_cur < (horz_offset + stride):
					histos[histo_offset * _codebook_size : (histo_offset + 1) * _codebook_size] += (
						horz_histos[horz_cur * _codebook_size: (horz_cur + 1) * _codebook_size])
					horz_cur += 1
				horz_offset += stride
				histo_offset += 1
			stride *= 2
		stride = 1
		while stride < vert_histos.shape[0] / _codebook_size:
			vert_offset = 0
			while vert_offset < vert_histos.shape[0] / _codebook_size:
				vert_cur = vert_offset
				while vert_cur < (vert_offset + stride):
					histos[histo_offset * _codebook_size : (histo_offset + 1) * _codebook_size] += (
						vert_histos[vert_cur * _codebook_size: (vert_cur + 1) * _codebook_size])
					vert_cur += 1
				vert_offset += stride
				histo_offset += 1
			stride *= 2

		#print
		#print "Whole Histo"
		#print histos

		# normalize
		for offset in xrange(_num_features / _codebook_size):
			s = histos[offset * _codebook_size : (offset + 1) * _codebook_size].sum()
			if s:
				histos[offset * _codebook_size : (offset + 1) * _codebook_size] /= float(s)
		self.features = histos

		#print
		#print "Normalized"
		#print self.features
		#exit()
		return self.features
		

def load_instances(in_dir):
	print "Loading Instances"
	if _read_cache and os.path.exists(_cache_instances_file):
		print "\tReading Cache"
		f = open(_cache_instances_file)
		try:
			instances = cPickle.load(f)
			f.close()
			print "Done\n"
			return instances
		except Exception as e:
			print "\tError loading instances:", e
			print "\tComputing From scratch"
			f.close()

	instances = list()
	for sdir in os.listdir(in_dir):
		if sdir not in ["UK1911Census_EnglandWales_Household15Names_03_01",
						"UK1911Census_EnglandWales_Household15Names_06_01"]:
			continue
		rdir = os.path.join(in_dir, sdir)
		for im_file in os.listdir(rdir):
			if im_file.endswith(_image_ext):
				label = sdir
				if label == "UK1911Census_EnglandWales_Household100Names_08_01":
					label = "UK1911Census_EnglandWales_Household40Names_07_01"
				instances.append(Instance(os.path.join(rdir, im_file), label))
	random.shuffle(instances)

	if _write_cache:
		print "\tWriting Instances to Cache"
		f = open(_cache_instances_file, 'w')
		try:
			cPickle.dump(instances, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()
	print "Done\n"
	return instances

def sample_surf_features(instances):
	print "Getting Surf Features for Codebook Construction"
	num_to_sample = int(len(instances) * _perc_docs_for_codebook)
	print "\tSampling Features from %d instances" % num_to_sample
	list_sampled_features = list()
	for x in xrange(num_to_sample):
		instances[x].calc_surf_features()
		list_sampled_features.append(instances[x].deses)
		if x % _print_interval == 0:
			print "\t\t%d/%d (%2.1f%%) Documents processed" % (x, num_to_sample, 100.0 * x / num_to_sample)
	for _ in list_sampled_features:
		print _
	sampled_features = np.concatenate(list_sampled_features)
	np.random.shuffle(sampled_features)
	return sampled_features[:_max_surf_features]

def construct_codebook(instances):
	print "Constructing Codebook"
	if _read_cache and os.path.exists(_cache_codebook_file):
		print "\tReading Cache"
		f = open(_cache_codebook_file)
		try:
			codebook = cPickle.load(f)
			f.close()
			print "Done\n"
			return codebook
		except Exception as e:
			print "\tError loading codebook:", e
			print "\tComputing From scratch"
			f.close()

	surf_feature_samples = sample_surf_features(instances)
	print "\tNumber of SURFs for codebook construction: ", surf_feature_samples.shape[0]

	if _use_k_medoids:
		print "\tComputing Distances"
		distances = scipy.spatial.distance.pdist(surf_feature_samples, 'cityblock')
		distances = scipy.spatial.distance.squareform(distances)
		print "\tDone\n"

		print "\tRunning Kmedoids"
		indices = kmedoids.cluster(distances, k=_codebook_size, maxIters=_max_k_medoids_iters)[1]
		codebook = surf_feature_samples[indices]
		print "\tDone\n"
	else:
		codebook = surf_feature_samples[:_codebook_size]

	if _write_cache:
		print "\tWriting Codebook to Cache"
		f = open(_cache_codebook_file, 'w')
		try:
			cPickle.dump(codebook, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()
	print "Done\n"
	return codebook

def compute_features(instances, codebook):
	print "Computing histogram features for each instance"
	if _read_cache and os.path.exists(_cache_data_matrix_file):
		print "\tReading Cache"
		f = open(_cache_data_matrix_file)
		try:
			features = cPickle.load(f)
			f.close()
			print "Done\n"
			return features
		except Exception as e:
			print "\tError loading Data Matrix:", e
			print "\tComputing From scratch"
			f.close()

	features = list()
	total = len(instances)
	for x, instance in enumerate(instances):
		features.append(instance.calc_features(codebook))
		if x % _print_interval == 0:
			print "\t\t%d/%d (%2.1f%%) Documents processed" % (x, total, 100.0 * x / total)
	features = np.array(features)
	#features = np.array(map(lambda instance: instance.calc_features(codebook), instances))

	if _write_cache:
		print "\tWriting Data Matrix to Cache"
		f = open(_cache_data_matrix_file, 'w')
		try:
			cPickle.dump(features, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()

	if _write_cache:
		print "\tWriting Instances to Cache"
		f = open(_cache_instances_file, 'w')
		try:
			cPickle.dump(instances, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()
	print "Done\n"

	print "Done\n"
	return features

def compute_random_matrix(data_matrix):
	print "Constructing Random Training Set"

	if _read_cache and os.path.exists(_cache_random_matrix_file):
		print "\tReading Cache"
		f = open(_cache_random_matrix_file)
		try:
			rand_mat = cPickle.load(f)
			f.close()
			print "Done\n"
			return rand_mat
		except Exception as e:
			print "\tError loading Random Matrix:", e
			print "\tComputing From scratch"
			f.close()

	rand_mat = np.zeros_like(data_matrix)
	for col in xrange(rand_mat.shape[1]):
		vals = data_matrix[:,col]
		for row in xrange(rand_mat.shape[0]):
			rand_mat[row, col] = np.random.choice(vals)

	if _write_cache:
		print "\tWriting Random Matrix to Cache"
		f = open(_cache_random_matrix_file, 'w')
		try:
			cPickle.dump(rand_mat, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()

	print "Done\n"
	return rand_mat

def train_classifier(real_data, fake_data):
	print "Training Random Forest"

	if _read_cache and os.path.exists(_cache_rf_file):
		print "\tReading Cache"
		f = open(_cache_rf_file)
		try:
			rf = cPickle.load(f)
			f.close()
			print "Done\n"
			return rf
		except Exception as e:
			print "\tError loading Random Forest:", e
			print "\tComputing From scratch"
			f.close()



	rf = sklearn.ensemble.RandomForestClassifier(n_estimators=_num_trees, max_features=_num_tree_features,
												bootstrap=False, n_jobs=6)
	combined_data = np.concatenate( (real_data, fake_data) )
	labels = np.concatenate( (np.ones(real_data.shape[0]), np.zeros(fake_data.shape[0])) )
	rf.fit(combined_data, labels)

	if _write_cache:
		print "\tWriting Random Forest to Cache"
		f = open(_cache_rf_file, 'w')
		try:
			cPickle.dump(rf, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()

	print "Done\n"
	return rf

def compute_sim_mat(data_matrix, random_forest):
	print "Computing the Similarity Matrix"
	if _read_cache and os.path.exists(_cache_sim_matrix_file):
		print "\tReading Cache"
		f = open(_cache_sim_matrix_file)
		try:
			sim_mat = cPickle.load(f)
			f.close()
			print "Done\n"
			return sim_mat
		except Exception as e:
			print "\tError loading Random Matrix:", e
			print "\tComputing From scratch"
			f.close()

	leaf_nodes = random_forest.apply(data_matrix)
	sim_mat = scipy.spatial.distance.pdist(leaf_nodes, "hamming")
	sim_mat = scipy.spatial.distance.squareform(sim_mat)
	sim_mat = 1 - sim_mat

	if _write_cache:
		print "\tWriting Similarity Matrix to Cache"
		f = open(_cache_sim_matrix_file, 'w')
		try:
			cPickle.dump(sim_mat, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()

	print "Done\n"
	return sim_mat

def spectral_cluster(affinity_matrix):
	print "Performing Spectral Clustering"

	if _read_cache and os.path.exists(_cache_cluster_assignments_file):
		print "\tReading Cache"
		f = open(_cache_cluster_assignments_file)
		try:
			assignments = cPickle.load(f)
			f.close()
			print "Done\n"
			return assignments
		except Exception as e:
			print "\tError loading Cluster Assignments:", e
			print "\tComputing From scratch"
			f.close()

	sc = sklearn.cluster.SpectralClustering(n_clusters=_number_of_clusters, affinity="precomputed",
											assign_labels="discretize")
	assignments = sc.fit_predict(affinity_matrix)

	if _write_cache:
		print "\tWriting Cluster Assignments to Cache"
		f = open(_cache_cluster_assignments_file, 'w')
		try:
			cPickle.dump(assignments, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()

	print "Done\n"
	return assignments

def form_clusters(instances, assignments):
	print "Forming Clusters"
	cluster_map = dict()
	class Mock:
		pass
	m = Mock()
	m.label = None
	for x in xrange(_number_of_clusters):
		cluster_map[x] = cluster.Cluster(list(), m, x)
	for instance, assignment in zip(instances, assignments):
		cluster_map[assignment].members.append(instance)
	clusters = cluster_map.values()
	map(lambda cluster: cluster.set_label(), clusters)
	print "Done\n"
	return clusters

def get_clusters(in_dir):
	instances = load_instances(in_dir)
	print "Num Instanes", len(instances)

	codebook = construct_codebook(instances)
	print "Codebook Shape", codebook.shape
	print codebook
	print

	data_matrix = compute_features(instances, codebook)
	print "Data Matrix Shape", data_matrix.shape
	print data_matrix
	print

	random_matrix = compute_random_matrix(data_matrix)
	print "Random Matrix Shape", random_matrix.shape
	print random_matrix
	print

	random_forest = train_classifier(data_matrix, random_matrix)
	sim_mat = compute_sim_mat(data_matrix, random_forest)
	print "Sim Matrix Shape", sim_mat.shape
	print sim_mat
	print

	cluster_assignments = spectral_cluster(sim_mat)
	print "Cluster Assignments"
	print cluster_assignments
	print

	clusters = form_clusters(instances, cluster_assignments) 

	for cluster in clusters:
		print cluster._id, cluster.label, collections.Counter(map(lambda instance: instance.label, cluster.members))

	return instances, clusters

def print_analysis(instances, clusters):
	class Mock:
		pass
	m = Mock()
	m.get_clusters = lambda: clusters
	m.get_docs = lambda: instances
	analyzer = metric.KnownClusterAnalyzer(m)
	analyzer.print_general_info()
	analyzer.print_histogram_info()
	analyzer.print_label_conf_mat()
	analyzer.print_label_cluster_mat()
	analyzer.print_label_info()
	analyzer.print_metric_info()

def main(in_dir):
	instances, clusters = get_clusters(in_dir)
	print_analysis(instances, clusters)


if __name__ == "__main__":
	in_dir = sys.argv[1]
	main(in_dir)
