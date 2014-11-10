
import datetime
import random
import shutil
import time
import sys
import os

import metric
import cluster
import utils
import lines
import doc

single_dir = "../data/wales100/UK1911Census_EnglandWales_Household15Names_03_01"
single_file = "rg14_31702_0069_03.txt"
second_dir = "../data/wales100/UK1911Census_EnglandWales_Household15Names_03_01"
second_file = "rg14_31704_0055_03.txt"
aggregate_dir = "../data/wales100/UK1911Census_EnglandWales_Household15Names_03_01"
#aggregate_dir = "../data/new/1911Wales/UK1911Census_EnglandWales_Household15Names_03_01"



def get_data_dir(descrip):
	if descrip.startswith("big"):
		return "../data/new/1911Wales"
		#return "../data/full/WashStatePassLists"
	if descrip.startswith("medium"):
		return "../data/wales1000/"
	if descrip.startswith("small"):
		return "../data/wales100/"
	if descrip.startswith("very_small"):
		return "../data/wales40/"
	if descrip.startswith("custom"):
		return "../data/walescustom/"
	if descrip.startswith("twoclass"):
		return "../data/walestwoclass/"

def cluster_known():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	random.seed(12345)
	random.shuffle(docs)
	sim_thresh = float(sys.argv[3])
	#confirm = cluster.BaseCONFIRM(docs, sim_thresh)
	#confirm = cluster.AdaptiveThresholdCONFIRM(docs, num_clust=5, num_instances=15, sim_thresh=sim_thresh)
	#confirm = cluster.BestCONFIRM(docs, lr=0.02, min_size=5, sim_thresh=sim_thresh)
	confirm = cluster.TestCONFIRM(docs, lr=0.02, min_size=10, sim_thresh=sim_thresh, num_clust=2, maxK=4)
	confirm.cluster()
	print
	print
	analyzer = metric.KnownClusterAnalyzer(confirm)
	analyzer.draw_centers()
	analyzer.print_all()

def check_init():
	docs = doc.get_docs_nested(get_data_dir("small"))
	random.seed(12345)
	random.shuffle(docs)
	confirm = cluster.MaxCliqueInitCONFIRM(docs, 7, 20)
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
	print
	print
	analyzer = metric.KnownClusterAnalyzer(confirm)
	analyzer.draw_centers()
	analyzer.print_all()

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
		if template is None:
			template = _doc
		else:
			template.aggregate(_doc)
			im = template.draw()
			im.save("output/aggregate/template_%d.png" % x)
			print
			print "IN DRIVER", id(template)
			for line in template.text_lines:
				print line
	template.final_prune()
	im = template.draw()
	im.save("output/aggregate/template_final.png")

def load_doc_test():
	_doc = doc.get_doc(single_dir, single_file)
	#_doc = doc.get_doc(second_dir, second_basename)
	_doc._load_check()
	for line in _doc.h_lines:
		print line
	for line in _doc.v_lines:
		print line
	for line in _doc.text_lines:
		print line
	im = _doc.draw()
	im.save("output/single_doc.png")

def cmp_test():
	doc1 = doc.get_doc(single_dir, single_file)
	doc2 = doc.get_doc(second_dir, second_file)
	doc1._load_check()
	doc2._load_check()

	#print "DOC1 H-lines:"
	#for line in doc1.h_lines:
	#	print "\t%s" % line
	#print

	#print "DOC2 H-lines:"
	#for line in doc2.h_lines:
	#	print "\t%s" % line
	#print

	sims = doc1.similarities_by_name(doc2)
	sim_mats = doc1.similarity_mats_by_name(doc2)
	for name in sim_mats:
		print name
		sim_mat = sim_mats[name]
		utils.print_mat(utils.apply_mat(sim_mat, lambda x: "%.3f" % x))
		print

	doc1.draw().save("output/doc1.png")
	doc2.draw().save("output/doc2.png")
	#print sims
	#print len(doc1.h_lines), len(doc1.v_lines)
	#print len(doc2.h_lines), len(doc2.v_lines)
	doc1.aggregate(doc2)
	doc1.draw().save("output/combined.png")
	#print len(doc1.h_lines), len(doc1.v_lines)

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
	


def main(arg):
	if arg == "cluster":
		cluster_known()
	if arg == "twice":
		double_cluster_known()
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
	if arg == "init":
		check_init()

if __name__ == "__main__":
	print "Start"
	print "Args: ", sys.argv
	start_time = time.time()
	main(sys.argv[1])
	end_time = time.time()
	print "End"
	print "Total Time elapsed: ", datetime.timedelta(seconds=(end_time - start_time))

