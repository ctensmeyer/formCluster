
import collections
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
import numpy as np


_output_dir = "output/"
#aggregate_dir = "../data/subsets/wales_100/UK1911Census_EnglandWales_Household15Names_06_01"
aggregate_dir = "../data/subsets/wales_100/UnClassified"

single_dir = "../data/subsets/wales_20/UK1911Census_EnglandWales_Household15Names_03_01"
single_file = "rg14_31702_0025_03.txt"
#second_dir = "../data/subsets/wales_100/UK1911Census_EnglandWales_Household40Names_07_01"
#second_file = "rg14_32127_0667_07.txt"

second_dir = single_dir
second_file = "rg14_31702_0059_03.txt"

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

	
	if descrip.startswith("wales_all"):
		return "../data/current/1911Wales"
	if descrip.startswith("wales_1000"):
		return "../data/subsets/wales_1000"
	if descrip.startswith("wales_500"):
		return "../data/subsets/wales_500"
	if descrip.startswith("wales_100"):
		return "../data/subsets/wales_100"
	if descrip.startswith("wales_20"):
		return "../data/subsets/wales_20"

	if descrip.startswith("wales_twoclass_all"):
		return "../data/subsets/wales_twoclass_all"
	if descrip.startswith("wales_twoclass_200"):
		return "../data/subsets/wales_twoclass_200"
	if descrip.startswith("wales_twoclass_100"):
		return "../data/subsets/wales_twoclass_100"

	if descrip.startswith("wash_all"):
		return "../data/current/WashStatePassLists"
	if descrip.startswith("wash_1000"):
		return "../data/subsets/wash_1000"
	if descrip.startswith("wash_500"):
		return "../data/subsets/wash_500"
	if descrip.startswith("wash_100"):
		return "../data/subsets/wash_100"
	if descrip.startswith("wash_20"):
		return "../data/subsets/wash_20"

	if descrip.startswith("nist_all"):
		return "../data/current/NIST"
	if descrip.startswith("nist_1000"):
		return "../data/subsets/nist_1000"
	if descrip.startswith("nist_500"):
		return "../data/subsets/nist_500"
	if descrip.startswith("nist_100"):
		return "../data/subsets/nist_100"
	if descrip.startswith("nist_20"):
		return "../data/subsets/nist_20"

	if descrip.startswith("padeaths_all"):
		return "../data/current/PADeaths"
	if descrip.startswith("padeaths_2000"):
		return "../data/subsets/padeaths_2000"
	if descrip.startswith("padeaths_1000"):
		return "../data/subsets/padeaths_1000"
	if descrip.startswith("padeaths_500"):
		return "../data/subsets/padeaths_500"
	if descrip.startswith("padeaths_100"):
		return "../data/subsets/padeaths_100"
	if descrip.startswith("padeaths_20"):
		return "../data/subsets/padeaths_20"

	if descrip.startswith("england_all"):
		return "../data/current/1911England"
	if descrip.startswith("england_5000"):
		return "../data/subsets/england_5000"
	if descrip.startswith("england_1000"):
		return "../data/subsets/england_1000"
	if descrip.startswith("england_500"):
		return "../data/subsets/england_500"
	if descrip.startswith("england_100"):
		return "../data/subsets/england_100"
	if descrip.startswith("england_20"):
		return "../data/subsets/england_20"

def get_confirm(descrip):
	if descrip == "base":
		return cluster.BaseTestCONFIRM
	if descrip == "region":
		return cluster.RegionTestCONFIRM
	if descrip == "weighted":
		return cluster.RegionWeightedTestCONFIRM
	if descrip == "wavg":
		return cluster.WavgNetTestCONFIRM

	if descrip == "best":
		return cluster.BestCONFIRM

	if descrip == "perfect_base":
		return cluster.PerfectCONFIRM
	if descrip == "perfect_region":
		return cluster.PerfectRegionCONFIRM
	if descrip == "perfect_weighted":
		return cluster.PerfectRegionWeightedCONFIRM
	if descrip == "perfect_wavg":
		return cluster.PerfectWavgNetCONFIRM

	if descrip == "kumar":
		return ncluster.BestKumarCONFIRM

	if descrip == "sskumar":
		return ncluster.SemiSupervisedKumarCONFIRM

	if descrip == "pipeline":
		return ncluster.PipelineCONFIRM

def cluster_known():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	random.shuffle(docs)
	param = int(sys.argv[3])
	param2 = int(sys.argv[5])
	param3 = int(sys.argv[6])

	factory = get_confirm(sys.argv[4])
	confirm = factory(docs, 
		sim_thresh=param,		 	# BaseCONFIRM
		num_instances=3, 			# InitCONFIRMs, how many instances to examine to find $num_clust clusters
		num_clust=2,  				# MaxCliqueCONFIRM, how many clusters try for
		lr=0.02,  					# learning rate WavgNet
		instances_per_cluster=10,  	# SupervisedInitCONFIRM, how many labeled instances start a cluster
		min_size=2, 				# PruningCONFIRM, clusters under this size get pruned
		maxK=4, 					# MaxClustersCONFIRM, max on created clusters (doesn't work)

		num_initial_seeds=param, 	# KumarCONFIRM, how many seeds to start with
		iterations=1,				# KumarCONFIRM, how many iterations to perform
		num_seeds=param,            # KumarCONFIRM, how many seeds to get each iteration
		cluster_range=(2,5),	 	# KumarCONFIRM, how many clusters to search over

		seeds_per_batch=2,  		# MaxCliqueSeedsKumarCONFIRM, how many seeds to get per batch
		batch_size=10,  			# MaxCliqueSeedsKumarCONFIRM, how many batches
		num_per_seed=param,			# SemiSupervisedKumarCONFIRM, how many docs/label to make seeds

		init_subset=30000,			# PipelineCONFIRM, how many docs to initialize
		min_membership=1, 			# PipelineCONFIRM, how many docs a cluster must have after initialization
		z_threshold=-100,			# PipelineCONFIRM, the reject threshold for the greedy pass
		use_labels=False,			# PipelineCONFIRM, Skips kumarconfirm init and uses the labels
		use_ss=param3
		)

	confirm.cluster_bootstrap()
	print
	print
	#exit()

	#if hasattr(confirm, 'print_reject_analysis'):
	#	confirm.print_reject_analysis()
	#elif hasattr(confirm, 'print_analysis'):
	#	confirm.print_analysis()
	#else:
	#	analyzer = metric.KnownClusterAnalyzer(confirm)
	#	analyzer.draw_centers()
	#	analyzer.print_all()

def extract():
	dataset = sys.argv[2]
	outdir = "output/" + "_".join(sys.argv[1:])

	docs = doc.get_docs_nested(get_data_dir(dataset))
	random.shuffle(docs)

	rand_amounts = [10, 20, 30, 50, 75, 100]
	type_percs = [0.01, 0.25, 0.50, 0.75, 0.90, 1.0]

	#rand_amounts = [1, 2, 3, 5]
	#type_percs = [0.01, 0.50]

	num_type_seeds = 30 if dataset not in ['nist', 'wales_balanced'] else 50
	#num_type_seeds = 7 if dataset not in ['nist', 'wales_balanced'] else 50

	extractor = ncluster.FeatureExtractor(docs)
	#extractor.extract_random(os.path.join(outdir, 'rand'), rand_amounts)
	extractor.extract_type(os.path.join(outdir, 'type'), num_type_seeds, type_percs)
	

def check_init():
	docs = doc.get_docs_nested(get_data_dir("test"))
	random.shuffle(docs)
	confirm = cluster.MaxCliqueInitCONFIRM(docs, 2, 10)
	confirm._init_clusters()

	print
	print "Cluster Sim Mat"
	sim_mat = confirm.get_cluster_sim_mat()
	utils.print_mat(utils.apply_mat(sim_mat, lambda x: "%3.2f" % x))



def double_cluster_known():
        docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
        epsilon = float(sys.argv[3])
        organizer = cluster.TemplateSorter(docs)
        organizer.go(epsilon)
        organizer.prune_clusters()
        clusters = organizer.get_clusters()
        print "Initial Clustering Complete"
        print "Reclustering..."
        centers = map(lambda x: x.center, clusters)
        organizer.go(epsilon,templates=centers)
        organizer.prune_clusters()
        clusters = organizer.get_clusters()
        print
        print
        analyzer = metric.KnownClusterAnalyzer(clusters)
        analyzer.draw_centers()
        analyzer.print_all()

def compare_true_templates():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	#confirm = cluster.PerfectCONFIRM(docs)
	confirm = cluster.BestPerfectCONFIRM(docs, lr=0.05)
	confirm.cluster()
	analyzer = metric.KnownClusterAnalyzer(confirm)
	analyzer.print_all()
	analyzer.draw_centers()
	analyzer.clusters[0].center.push_away(analyzer.clusters[1].center)
	print "PUSHING APART!"
	print
	print
	analyzer = metric.KnownClusterAnalyzer(confirm)
	analyzer.draw_centers()
	analyzer.print_all()
	print
	print

def aggreage_same():
	docs = doc.get_docs(aggregate_dir)[0]
	try:
		shutil.rmtree('output/aggregate')
	except:
		pass
	try:
		os.mkdir('output/aggregate')
	except:
		pass
	for x, _doc in enumerate(docs):
		_doc._load_check()
		#im = _doc.draw()
		#im.save("output/aggregate/doc_%d.png" % x)
	template = None
	for x, _doc in enumerate(docs):
		print
		print "************* Adding in doc %d ********************" % x
		print _doc._id
		print
		#im = _doc.draw()
		#im.save("output/aggregate/doc_%d.png" % x)

		if template is None:
			template = _doc
		else:
			template.aggregate(_doc)
			#im = template.draw()
			#im.save("output/aggregate/template_%d.png" % x)
	template.final_prune()
	im = template.draw()
	im.save("output/aggregate/template_final.png")

def load_doc_test():
	_doc = doc.get_doc(single_dir, single_file)
	#_doc = doc.get_doc(second_dir, second_basename)
	_doc._load_check()
	_doc.display()
	im = _doc.draw()
	im.save("output/single_doc.png")

def cmp_test():
	doc1 = doc.get_doc(single_dir, single_file)
	doc2 = doc.get_doc(second_dir, second_file)
	doc1._load_check()
	doc2._load_check()

	doc1.display()
	doc2.display()

	global_region_sims = doc1.global_region_sim(doc2)
	global_region_weights = doc1.global_region_weights()
	global_sims = doc1.global_sim(doc2)
	region_sims = doc1.region_sim(doc2)
	region_weights1 = doc1.region_weights()
	region_weights2 = doc2.region_weights()


	for x, name in enumerate(doc1.feature_set_names):
		print
		print name
		print "Global Sim:", global_sims[x]
		print "Region Sims:"
		print
		utils.print_mat(utils.apply_mat(region_sims[x], lambda x: "%.3f" % x))
		print
		print "Region Weights doc1:"
		print
		utils.print_mat(utils.apply_mat(region_weights1[x], lambda x: "%.3f" % x))
		print
		print "Region Weights doc2:"
		print
		utils.print_mat(utils.apply_mat(region_weights2[x], lambda x: "%.3f" % x))
	print
	print "Match Vec"
	print
	match_vec = doc1.match_vector(doc2)
	for x in xrange(len(match_vec) / 10):
		print match_vec[10 * x: 10 * (x + 1)]
	print

	print "Sim Vector:"
	print " ".join(map(lambda x: "%.2f" % x, global_region_sims))
	print "Sim Weights:"
	print " ".join(map(lambda x: "%.2f" % x, global_region_weights))

	#doc1.draw().save("output/doc1.png")
	#doc2.draw().save("output/doc2.png")

	#doc1.push_away(doc2)
	#doc1.draw().save("output/doc1_pushed.png")
	#doc2.draw().save("output/doc2_pushed.png")

	#doc1.push_away(doc2)

	doc1.aggregate(doc2)
	doc1.display()
	#doc1.draw().save("output/combined.png")

def draw_all():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	try:
		shutil.rmtree('output/docs')
	except:
		pass
	try:
		os.mkdir('output/docs')
	except:
		pass
	for _doc in docs:
		_doc.draw().save("output/docs/%s.png" % _doc._id)
	

def test_features_syn():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	max_size = int(sys.argv[3])
	num_combine = int(sys.argv[4])
	min_size = int(sys.argv[5])

	d = collections.defaultdict(list)
	for _doc in docs:
		d[_doc.label].append(_doc)
	pure_clusters = d.values()
	broken_clusters = list()
	for x in xrange(10):
		for _cluster in pure_clusters:
			broken_clusters += [_cluster[i:i + max_size] for i in range(0, len(_cluster), max_size)]
		combined_clusters = list()
		while broken_clusters:
			if len(broken_clusters) < num_combine:
				clusters = list(broken_clusters)
			else:
				clusters = random.sample(broken_clusters, num_combine)
			for _cluster in clusters:
				broken_clusters.remove(_cluster)
			combined_clusters.append(utils.flatten(clusters))

		clusters = map(lambda combined_cluster: cluster.Cluster(combined_cluster), combined_clusters)
		ncluster.test_features(clusters, min_size)

def test_features():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	random.shuffle(docs)
	ncluster.test_splitting(docs)

def all_cluster():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	num_subset = int(sys.argv[3])
	num_initial_clusters = int(sys.argv[4])
	num_seeds = int(sys.argv[5])
	min_pts = int(sys.argv[6])
	outdir = os.path.join(_output_dir, str(datetime.date.today()) + "_" + "_".join(sys.argv[1:]))
	ncluster.all_cluster(docs, num_subset, num_initial_clusters, num_seeds, min_pts, outdir)

def overall_experiment():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	num_types = len(set(map(lambda _doc: _doc.label, docs)))
	num_subset = len(docs)
	num_seeds = 10
	#initial_cluster_range = range(num_types / 2, int(1.5 * num_types))
	initial_cluster_range = [10]
	min_pts = 5
	ncluster.overall(docs, num_subset, num_seeds, initial_cluster_range, min_pts)
	#ncluster.test_par(docs, num_subset, num_seeds, initial_cluster_range, min_pts)

def test_par():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	num_seeds = 5
	seeds = random.sample(docs, num_seeds)

	# parallel
	start_time = time.time()
	features_par = ncluster.extract_features_par(docs, seeds)[0]
	end_time = time.time()
	print "Parallel Time elapsed: ", datetime.timedelta(seconds=(end_time - start_time))

	# serial
	start_time = time.time()
	features_ser = ncluster.extract_features(docs, seeds)[0]
	end_time = time.time()
	print "Serial Time elapsed: ", datetime.timedelta(seconds=(end_time - start_time))

	for x in xrange(features_par.shape[0]):
		for y in xrange(features_par.shape[1]):
			if features_par[x,y] != features_ser[x,y]:
				print x, y, features_par[x,y], features_ser[x,y]


def test():
	docs = doc.get_docs_nested(get_data_dir("wales_20"))
	num_types = len(set(map(lambda _doc: _doc.label, docs)))
	num_subset = len(docs)
	num_seeds = 4
	initial_cluster_range = [3, 4, 5]
	min_pts = 2
	ncluster.overall(docs, num_subset, num_seeds, initial_cluster_range, min_pts)

def subset_experiment():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))

	num_types = len(set(map(lambda _doc: _doc.label, docs)))
	print "Num Types:", num_types
	initial_cluster_range = list()
	if num_types != 2:
		initial_cluster_range.append(num_types / 2)
	initial_cluster_range.append(num_types)
	initial_cluster_range.append(int(1.5 * num_types))

	possible_subsets = [100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
	subsets = list()
	for s in possible_subsets:
		if s < len(docs):
			subsets.append(s)
	subsets.append(len(docs))
	for num_subset in subsets:
		num_seeds = 50
		min_pts = 30
		ncluster.overall(docs, num_subset, num_seeds, initial_cluster_range, min_pts)

def subset_experiment2():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))

	num_types = len(set(map(lambda _doc: _doc.label, docs)))
	print "Num Types:", num_types
	initial_cluster_range = list()
	if num_types != 2:
		initial_cluster_range.append(num_types / 2)
	initial_cluster_range.append(num_types)
	initial_cluster_range.append(int(1.5 * num_types))

	subsets = [int(sys.argv[3])]
	for num_subset in subsets:
		num_seeds = 50
		min_pts = 30
		ncluster.overall(docs, num_subset, num_seeds, initial_cluster_range, min_pts)

def run():
	try:
		# sys.argv[3] is the number of threads
		Ks = map(int, sys.argv[4].split(","))
		subsets = map(int, sys.argv[5].split(","))
		seeds = map(int, sys.argv[6].split(","))
		docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
		
	except:
		print "python driver.py run dataset #threads Ks subsets seeds"
		return

	Ks.sort()
	subsets.sort()
	seeds.sort()
	filtered = filter(lambda x: x < len(docs), subsets)
	if len(filtered) < len(subsets):
		filtered.append(len(docs))
		subsets = filtered

	ncluster.run_no_split(docs, Ks, subsets, seeds)

def auto():
	try:
		# sys.argv[3] is the number of threads
		Ks = map(int, sys.argv[4].split(","))
		subsets = map(int, sys.argv[5].split(","))
		seeds = map(int, sys.argv[6].split(","))
		docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
		
	except:
		print "python driver.py auto dataset #threads Ks subsets seeds"
		return

	Ks.sort()
	subsets.sort()
	seeds.sort()
	filtered = filter(lambda x: x < len(docs), subsets)
	if len(filtered) < len(subsets):
		filtered.append(len(docs))
		subsets = filtered

	ncluster.run_auto_minpts(docs, Ks, subsets, seeds)

def type():
	try:
		# sys.argv[3] is the number of threads
		Ks = map(int, sys.argv[4].split(","))
		subsets = map(int, sys.argv[5].split(","))
		seeds = map(int, sys.argv[6].split(","))
		types = map(int, sys.argv[7].split(","))
		docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
		
	except:
		print "python driver.py type dataset #threads Ks subsets seeds types"
		return

	Ks.sort()
	subsets.sort()
	seeds.sort()
	filtered = filter(lambda x: x < len(docs), subsets)
	if len(filtered) < len(subsets):
		filtered.append(len(docs))
		subsets = filtered

	ncluster.run_type(docs, Ks, subsets, seeds, types)
	

def extract():
	try:
		# sys.argv[3] is the number of threads
		num_seeds = int(sys.argv[4])
		feature_file = sys.argv[5]
		manifest_file = sys.argv[6]
		docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	except:
		print "python driver.py extract dataset #threads num_seeds feature_file manifest_file"
		return

	seeds = random.sample(docs, min(num_seeds, len(docs)))
	feature_mat = ncluster.extract_features_par(docs, seeds)
	np.save(feature_file, feature_mat)
	out = open(manifest_file, 'w')
	for _doc in docs:
		out.write("%s\n" % _doc.source_file)

	out.close()
	

def main(arg):
	if arg == "cluster":
		cluster_known()
	if arg == "perfect":
		compare_true_templates()
	if arg == "single":
		load_doc_test()
	if arg == "double":
		cmp_test()
	if arg == "aggregate":
		aggreage_same()
	if arg == "draw":
		draw_all()
	if arg == "run":
		run()
	if arg == "extract":
		extract()
	if arg == "auto":
		auto()
	if arg == "type":
		type()

if __name__ == "__main__":
	print "Start"
	print "Args: ", sys.argv
	start_time = time.time()
	main(sys.argv[1])
	end_time = time.time()
	print "End"
	print "Total Time elapsed: ", datetime.timedelta(seconds=(end_time - start_time))

