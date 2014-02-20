
import sys
import os
import compare
from doc import Document
import time
import hac
import cluster_metric as cm


_ocr_dir = "/home/chris/Ancestry/Data/small/1911England/OCR/"
_image_dir = "/home/chris/Ancestry/Data/small/1911England/images/"
_num_docs = 26

#_big_data_dir = "/home/chris/Ancestry/Data/large/1911Wales/"
_big_data_dir = "/home/chris/Ancestry/Data/test1500/"

def is_ocr_file(filename):
	if not filename.endswith('.xml'):
		return False
	if 'line' in filename:
		return False
	return True

def create_all_docs_large():
	docs = []
	num_loaded = 0
	num_exceptions = 0
	for _dir in os.listdir(_big_data_dir):
		print "Starting Dir: ", _dir
		r_dir = os.path.join(_big_data_dir, _dir)
		basenames = set()
		image_names = os.listdir(r_dir)
		for name in image_names:
			if name.endswith(".jpg"):
				basename = os.path.splitext(name)[0]
				basenames.add(basename)
		for basename in basenames:
			try:
				ocr_name = basename + ".xml"
				prof_name = basename + "_line.xml"
				form_name = basename + "_FormType.txt"
				image_name = basename + ".jpg"
				image_path = os.path.join(r_dir, image_name)
				ocr_path = os.path.join(r_dir, ocr_name)
				prof_path = os.path.join(r_dir, prof_name)
				form_path = os.path.join(r_dir, form_name)
				#exists = map(lambda path: os.path.exists(path), [image_path, prof_path, form_path, ocr_path])
				#if not all(exists):
				#	for path in [image_path, prof_path, form_path, ocr_path]:
				#		open(path)  # force exception
				doc = Document(basename, image_path, ocr_path, prof_path, form_path)
				docs.append(doc)
				num_loaded += 1
				if num_loaded % 10 == 0:
					print "Loaded %d documents" % num_loaded
			except:
				print "Exception reading ", os.path.join(_dir, basename)
				num_exceptions += 1
	print "%d Docs read" % num_loaded
	print "%d Docs could not be read" % num_exceptions
	return docs
	

def write_to_examine(outfile, data):
	'''
	Write out data in row/col format
	:param outfile: str filename
	:param data: list of tuples. Each tuple is a row.
	'''
	with open(outfile, 'w') as out:
		row = " " * 4 + " ".join(map(lambda n: "{:<5d}".format(n), xrange(1, _num_docs + 1))) + "\n"
		out.write(row)
		for x in xrange(len(data)):
			# this is why I love python
			row = "%3d %s\n" % (x + 1, " ".join(map(lambda n: "%5.3f" % ((-1 * n) + 1) if n else " " * 5, data[x])))
			out.write(row)
		out.flush()
		out.close()

def write_to_cluster(outfile, data):
	'''
	Write out data in row/col format
	:param outfile: str filename
	:param data: list of tuples. Each tuple is a row.
	'''
	with open(outfile, 'w') as out:
		for x in xrange(len(data)):
			# this is why I love python
			row = "%d %s\n" % (x, " ".join(map(lambda n: "%5.3f" % n, data[x])))
			out.write(row)
		out.flush()
		out.close()

def dist_func_gen(docs, mat):
	inverse = dict()
	for x, doc in enumerate(docs):
		inverse[doc._id] = x
	def dist_func(doc1, doc2):
		return mat[inverse[doc1._id]][inverse[doc2._id]]
	return dist_func
	
def get_all_labels(docs):
	labels = set()
	for doc in docs:
		labels.add(doc.label)
	return labels

def main():
	print "Start"
	start_time = time.time()
	docs = create_all_docs_large()
	load_time = time.time()
	print "Seconds to load docs:", (load_time - start_time)
	print "Docs per second:", len(docs) / float(load_time - start_time) 
	mat = []
	for x, d1 in enumerate(docs):
		row = []
		for y, d2 in enumerate(docs):
			if x == y:
				row.append(0)
			elif y < x:
				row.append(mat[y][x])
			else:
				#print
				#print "Comparing Documents %d and %d:" % (x + 1, y + 1)
				row.append(1 - compare.compare(d1, d2)) # convert similarity to difference
		mat.append(row)
		if len(mat) % 10 == 0:
			print "%d rows done!" % len(mat)
	mat_time = time.time()
	print "Done Calculating Distance Matrix"
	print "Seconds to calc Matrix:", (mat_time - load_time)
	dist_func = dist_func_gen(docs, mat)
	_hac = hac.HAC(docs, dist_func, hac.average, False)
	cluster_sets = _hac.cluster([0, 20])
	cluster_time = time.time()
	print "Done Clustering"
	print "Seconds to cluster:", (cluster_time - mat_time)
	for k, cluster_set in cluster_sets.iteritems():
		print k, cm.accuracy(cluster_set, cm.majority_labels(cluster_set))
	
	print "Total Time:", (time.time() - start_time)
	

if __name__ == "__main__":
	main()
	
