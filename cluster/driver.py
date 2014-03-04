
import time
import datetime
import metric
import cluster
import utils
import sys
import doc

#data_dir = "../data/lines/1911Wales/"
data_dir = "../data/wales100/"
epsilon = float(sys.argv[1])


def cluster_known():
	docs = doc.get_docs(data_dir)
	organizer = cluster.TemplateSorter(docs)
	organizer.go(epsilon)
	clusters = organizer.get_clusters()
	print
	print
	analyzer = metric.KnownClusterAnalyzer(clusters)
	analyzer.print_all()

def compare_true_templates():
	docs = doc.get_docs(data_dir)
	organizer = cluster.CheatingSorter(docs)
	organizer.go()
	


def main():
	cluster_known()

if __name__ == "__main__":
	print "Start"
	start_time = time.time()
	main()
	end_time = time.time()
	print "End"
	#print "%f seconds" % (end_time - start_time)
	print "Total Time elapsed: ", datetime.timedelta(seconds=(end_time - start_time))
