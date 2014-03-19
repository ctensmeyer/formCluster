
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
data_dir = "../data/wales40/"
single_dir = "../data/wales100/UK1911Census_EnglandWales_Household15Names_03_01"
single_basename = "rg14_31703_0073_03"
second_dir = "../data/wales100/UK1911Census_EnglandWales_Household15Names_03_01"
second_basename = "rg14_31703_0259_03"


def cluster_known():
	epsilon = float(sys.argv[1])
	docs = doc.get_docs_nested(data_dir)
	organizer = cluster.TemplateSorter(docs)
	organizer.go(epsilon)
	#organizer.prune_clusters()
	clusters = organizer.get_clusters()
	print
	print
	analyzer = metric.KnownClusterAnalyzer(clusters)
	analyzer.print_all()

def compare_true_templates():
	docs = doc.get_docs_nested(data_dir)
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

def cmp_test():
	doc1 = doc.get_doc(single_dir, single_basename)
	doc2 = doc.get_doc(second_dir, second_basename)
	h_sim = doc1.h_line_similarity(doc2)
	v_sim = doc1.v_line_similarity(doc2)
	print "H sim:", h_sim
	print "V sim:", v_sim
	lines.draw_lines(doc1.h_lines, doc1.v_lines, "output/doc1_lines.png", doc1.size)
	lines.draw_lines(doc2.h_lines, doc2.v_lines, "output/doc2_lines.png", doc2.size)

def main():
	cluster_known()
	#compare_true_templates()
	#load_doc_test()
	#cmp_test()

if __name__ == "__main__":
	print "Start"
	start_time = time.time()
	main()
	end_time = time.time()
	print "End"
	print "Total Time elapsed: ", datetime.timedelta(seconds=(end_time - start_time))

