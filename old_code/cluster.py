
import os
import sys
import shutil
import ast

_extension = '.png'
#HAC

def dists(clust1, clust2, index):
	dists = []
	for idx1 in clust1:
		for idx2 in clust2:
			dists.append(index[idx1]['vals'][idx2])
	return dists

def shortest(dists):
	return min(dists)

def longest(dists):
	return max(dists)

def average(dists):
	return sum(dists) / float(len(dists))

def init_clusters(infile):
	index = {}
	clusters = []
	with open(infile, 'r') as input:
		for line in input.readlines():
			tokens = line.split()
			idx = int(tokens[0])
			vals = map(ast.literal_eval, tokens[1:])
			index[idx] = {'vals': vals}
			clusters.append( (idx,) )

	return index, clusters


def main(infile, cluster_idx, indir, outdir):
	index, clusters = init_clusters(infile)
	distance_metric = average

	# combine n clusters n-1 times
	for i in xrange(len(clusters) - 1):
		
		# identity closest cluster according to distance_metric
		min_metric = 2.0 ** 20
		one = two = None
		for x, clust1 in enumerate(clusters):
			for y, clust2 in enumerate(clusters):
				if y == x:
					continue
				metric = distance_metric(dists(clust1, clust2, index))
				if metric < min_metric:
					min_metric = metric
					one = clust1
					two = clust2

		# combine clusters one and two
		#print "%d: %f\t %s %s" % (i, min_metric, one, two)
		combined = one + two
		clusters.remove(one)
		clusters.remove(two)
		clusters.append(combined)
		if cluster_idx is not None:
			if len(clusters) == cluster_idx:
				print "%d\t%f\t" % (len(clusters), min_metric), clusters
				if indir and outdir:
					try:
						os.mkdir(outdir)
					except:
						pass
					for d in os.listdir(outdir):
						try:
							shutil.rmtree(os.path.join(outdir, d))
						except:
							pass
					for cluster_num, cluster in enumerate(clusters, 1):
						cluster_path = os.path.join(outdir, str(cluster_num))
						os.mkdir(cluster_path)
						for idx in cluster:
							# copy image file from indir to outdir
							#filename = index[idx]['file'].split('.')[0] + _extension
							filename = ("im%d" % (idx + 1)) + _extension
							#try:
							shutil.copy(os.path.join(indir, filename), cluster_path)
							#except:
							#	print "Could not copy filename to %s" % cluster_path
		else:
			print clusters



if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise Exception("sim_mat_file, num_clusters, image_input_dir, image_output_dir")
		
	infile = sys.argv[1]
	cluster_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
	indir = sys.argv[3] if len(sys.argv) > 3 else None
	outdir = sys.argv[4] if len(sys.argv) > 4 else None
	main(infile, cluster_idx, indir, outdir)

