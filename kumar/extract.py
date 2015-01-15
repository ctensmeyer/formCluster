
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
_surf_threshold = 30000
_num_surf_features = 10000

# Codebook & Features
_codebook_sizes = [100, 200, 300, 400, 500]
_num_trials = 10
_perc_docs_for_codebook = 0.05
_num_surf_features_codebook = 30000
_max_k_medoids_iters = 30
_H_partitions = 4
_V_partitions = 3

# not parameters
_num_histograms = ( (2 ** (_H_partitions) - 1) + (2 ** (_V_partitions) - 1) - 1 )
_surf_instance = cv2.SURF(_surf_threshold)
_surf_instance.upright = _surf_upright
_surf_instance.extended = _surf_extended
_surf_features = dict()
_print_interval = 20


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
		deses = deses[:_num_surf_features] + 0
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
	num_to_sample = int(len(instances) * _perc_docs_for_codebook)
	sampled_instances = random.sample(instances, num_to_sample)
	list_sampled_features = list()
	for instance in sampled_instances:
		deses = calc_surf_features(instance[0])[1]
		list_sampled_features.append(deses)
	sampled_features = np.concatenate(list_sampled_features)
	np.random.shuffle(sampled_features)
	return sampled_features[:_num_surf_features_codebook]

def construct_codebook(instances, codebook_size):

	surf_feature_samples = sample_surf_features(instances)

	distances = scipy.spatial.distance.pdist(surf_feature_samples, 'cityblock')
	distances = scipy.spatial.distance.squareform(distances)

	indices = kmedoids.cluster(distances, k=codebook_size, maxIters=_max_k_medoids_iters)[1]
	codebook = surf_feature_samples[indices]

	return codebook

def compute_features(instances, codebook):

	features = list()
	total = len(instances)
	for instance in instances:
		features.append(calc_features(instance[0], codebook))
	features = np.array(features)

	return features


def main(in_dir, out_dir):
	try:
		os.makedirs(out_dir)
	except:
		pass
	instances = load_instances(in_dir)
	true_labels = np.array(map(lambda tup: tup[1], instances))
	np.save(os.path.join(out_dir, "labels.npy"), true_labels)
	print "Num Instanes", len(instances)

	for x, instance in enumerate(instances):
		calc_surf_features(instance[0])
		if x % _print_interval == 0:
			print "\t%d/%d (%2.1f%%) Images processed" % (x, len(instances), 100.0 * x / len(instances))

	codebook_params = list()
	for codebook_size in _codebook_sizes:
		for trial in xrange(_num_trials):
			codebook_params.append(codebook_size)
			
	num_codebooks = len(codebook_params)
	print
	print "Num Codebooks:", num_codebooks
	print codebook_params
	total_trials = len(_codebook_sizes) * _num_trials
	for y, codebook_size in enumerate(codebook_params):
		print "\t%d/%d (%2.1f%%) Codebooks processed" % (y, num_codebooks, 100.0 * y / num_codebooks)

		codebook = None
		while codebook is None:
			try:
				codebook = construct_codebook(instances, codebook_size)
			except:
				print "Creating Codebook failed, trying again"
		data_matrix = compute_features(instances, codebook)
		np.save(os.path.join(out_dir, "data_matrix_%d.npy" % y), data_matrix)


if __name__ == "__main__":
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	main(in_dir, out_dir)

