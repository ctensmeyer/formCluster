
import os
import sys
import collections

# first is the x-axis
_keys = ['', 'Num_Clusters'] 
_values = ['Acc', 'V-measure', 'Completeness', 'Homogeneity', 'ARI', 'Silhouette']


def mean(l):
	return sum(l) / len(l)

def process_dict(d):
	e = dict()
	for key in d:
		tups = d[key]
		avgs = map(lambda x: sum(map(lambda tup: tup[x], tups)) / len(tups), xrange(len(tups[0])))
		e[key] = avgs
	return e

def get_values(matrix, headers, keys, values):
	d = collections.defaultdict(list)
	key_idxs = map(lambda name: col_name_to_idx(headers, name), keys)
	value_idxs = map(lambda name: col_name_to_idx(headers, name), values)
	for row in matrix:
		key = tuple([row[x] for x in key_idxs])
		value = tuple([row[x] for x in value_idxs])
		d[key].append(value)
	return d

def load_file(infile):
	f = open(infile, 'r')
	headers = f.readline().strip().split("\t")
	matrix = list()
	for line in f.readlines():
		row = map(float, line.strip().split("\t"))
		matrix.append(row)
	return headers, matrix

def col_name_to_idx(headers, name):
	return headers.index(name)

def write(d, outdir):
	try:
		os.makedirs(outdir)
	except:
		pass
	
	lines = collections.defaultdict(list)
	for key in d:
		x, fd = key 
		vals = d[key]
		line = (x, "\t".join(map(lambda y: "%.3f" % y, vals)) )
		lines[fd].append(line)

	for fd in lines:
		f = open(os.path.join(outdir, "%s_%d.txt" % (_keys[1], int(fd))), 'w')
		tups = lines[fd]
		tups.sort(key=lambda tup: tup[0])
		for tup in tups:
			f.write("%.2f\t%s\n" % tup)
		f.close()
		
			

def main(infile, outdir):
	headers, matrix = load_file(infile)
	d = get_values(matrix, headers, _keys, _values)
	d = process_dict(d)
	write(d, outdir)


if __name__ == "__main__":
	infile = sys.argv[1]
	outdir = sys.argv[2]
	main(infile, outdir)
