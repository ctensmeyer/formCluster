
import time
import metric
import cluster
import utils
import sys
import doc

#data_dir = "/home/chris/Ancestry/Data/wales1000/"
data_dir = "../data/toy40/"
epsilon = float(sys.argv[1])

def main():
	docs = doc.get_docs(data_dir)
	organizer = cluster.TemplateSorter(docs)
	organizer.go(epsilon)
	clusters = organizer.get_clusters()
	print
	print
	analyzer = metric.ClusterAnalyzer(clusters)
	analyzer.print_all()


if __name__ == "__main__":
	print "Start"
	start_time = time.time()
	main()
	end_time = time.time()
	print "End"
	print "%f seconds" % (end_time - start_time)
	
