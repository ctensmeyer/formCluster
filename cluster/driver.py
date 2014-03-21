
import datetime
import time
import sys
import os

import metric
import cluster
import utils
import lines
import doc

#data_dir = "../data/lines/1911Wales/"
data_dir = "../data/wales100/"
single_dir = "../data/wales100/UK1911Census_EnglandWales_Household15Names_03_01"
single_basename = "rg14_31703_0073_03"
second_dir = "../data/wales100/UK1911Census_EnglandWales_Household15Names_03_01"
second_basename = "rg14_31703_0259_03"


def get_data_dir(descrip):
	if descrip == "big":
		return "../data/lines/1911Wales"
	if descrip == "medium":
		return "../data/wales1000/"
	if descrip == "small":
		return "../data/wales100/"

def cluster_known():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	epsilon = float(sys.argv[3])
	organizer = cluster.TemplateSorter(docs)
	organizer.go(epsilon)
	#organizer.prune_clusters()
	clusters = organizer.get_clusters()
	print
	print
	analyzer = metric.KnownClusterAnalyzer(clusters)
	analyzer.print_all()

def compare_true_templates():
	docs = doc.get_docs_nested(get_data_dir(sys.argv[2]))
	organizer = cluster.CheatingSorter(docs)
	organizer.go()
	clusters = organizer.get_clusters()
	print
	print
	analyzer = metric.KnownClusterAnalyzer(clusters)
	analyzer.print_all()

def load_doc_test():
	_doc = doc.get_doc(single_dir, single_basename)
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
	doc1 = doc.get_doc(single_dir, single_basename)
	doc2 = doc.get_doc(second_dir, second_basename)
	doc1.draw().save("output/doc1.png")
	doc2.draw().save("output/doc2.png")
	sims = doc1.similarities_by_name(doc2)
	print sims
	print len(doc1.h_lines), len(doc1.v_lines)
	print len(doc2.h_lines), len(doc2.v_lines)
	doc1.aggregate(doc2)
	doc1.draw().save("output/combined.png")
	print len(doc1.h_lines), len(doc1.v_lines)


def main(arg):
	if arg == "cluster":
		cluster_known()
	if arg == "perfect":
		compare_true_templates()
	if arg = "single":
		load_doc_test()
	if arg = "double":
		cmp_test()

if __name__ == "__main__":
	print "Start"
	print "Args: " + sys.argv
	start_time = time.time()
	main(sys.argv[1])
	end_time = time.time()
	print "End"
	print "Total Time elapsed: ", datetime.timedelta(seconds=(end_time - start_time))

