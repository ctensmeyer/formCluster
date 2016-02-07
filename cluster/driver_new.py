
import collections
import argparse
import datetime
import random
import shutil
import time
import sys
import os

import ncluster
import cluster
import metric
import utils
import lines
import doc
from constants import *

def get_data_dir(descrip):
	if descrip.startswith("wales_large"):
		return "../data/paper/Wales-Large"
	if descrip.startswith("wales_small"):
		return "../data/paper/Wales-Small"
	if descrip.startswith("wales_balanced"):
		return "../data/paper/Wales-Balanced"
	if descrip.startswith("washpass"):
		return "../data/paper/WashPass"
	if descrip.startswith("nist"):
		return "../data/paper/Nist"
	if descrip.startswith("padeaths_all"):
		return "../data/paper/PADeaths"
	if descrip.startswith("padeaths_balanced"):
		return "../data/paper/PADeaths-Balanced"

def parse_args():
	parser = argparse.ArgumentParser(description='CONFIRM')
	
	parser.add_argument('dataset', type=str, help='tag for dataset to run')
	parser.add_argument('-k', '--num-clusters', type=str, default='2',
			help='comma separated list of num clusters for initial clustering')
	parser.add_argument('-e', '--num-exemplars', type=str, default='2',
			help='comma separated list of num exemplars for initial clustering')
	parser.add_argument('-s', '--subset-sizes', type=str, default='2',
			help='comma separated list of size of initially clustered subsets')

	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('-t', '--num-types', type=str, default='2',
			help='comma separated list of number of oracle types for exemplar selection')
	group.add_argument('-r', '--rand-exemplars', default=False, action='store_true',
			help='comma separated list of number of oracle types for exemplar selection')

	parser.add_argument('--euclidean', default=False, action='store_true',
			help='Use Euclidean distance between feature vectors for Spectral Clustering')
	parser.add_argument('--init-only', default=False, action='store_true',
			help='Only run the initial clustering')
	parser.add_argument('--no-refine', default=False, action='store_true',
			help='Skip the cluster refinement step')

	parser.add_argument('--text-only', default=False, action='store_true',
			help='Only use Text page elements')
	parser.add_argument('--rule-only', default=False, action='store_true',
			help='Only use Rule Line page elements')

	group = parser.add_argument_group()
	group.add_argument('--no-auto-minpts', default=False, action='store_true',
			help='Do not set min_pts adaptively for OPTICS cluster refinement')
	group.add_argument('--minpts', type=int, default=30,
			help='min_pts for OPTICS cluster refinement.  Min value unless --no-auto-minpts is set.')
	group.add_argument('--minpts-perc', type=float, default=0.1,
			help='For adaptive min_pts, multiplier for the cluster size')

	args = parser.parse_args()
	return args

def process_args(args):
	Ks = map(int, args.num_clusters.split(","))
	Ks.sort()
		
	num_exemplars = map(int, args.num_exemplars.split(","))
	num_exemplars.sort()

	subset_sizes = map(int, args.subset_sizes.split(","))
	subset_sizes.sort()

	num_types =  map(int, args.num_types.split(",")) if not args.rand_exemplars else []
	num_types.sort()

	docs = doc.get_docs_nested(get_data_dir(args.dataset))
	random.shuffle(docs)
	num_docs = len(docs)

	subset_sizes = filter(lambda x: x >= 2 and x <= num_docs, subset_sizes)
	smallest_subset = min(subset_sizes)

	Ks = filter(lambda x: x >= 2 and x <= smallest_subset, Ks)
	num_exemplars = filter(lambda x: x >= 2 and x <= smallest_subset, num_exemplars)

	return docs, Ks, subset_sizes, num_exemplars, num_types

def main():
	args = parse_args()
	docs, Ks, subset_sizes, num_exemplars, num_types = process_args(args)
	ncluster.confirm(docs, Ks, subset_sizes, num_exemplars, num_types, args)


if __name__ == "__main__":
	print "Start"
	print "Args: ", sys.argv
	start_time = time.time()
	main()
	end_time = time.time()
	print "End"
	print "Total Time elapsed: ", datetime.timedelta(seconds=(end_time - start_time))
