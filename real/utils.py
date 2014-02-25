
import os
import doc
import math

def e_dist(p1, p2):
	return math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 )

def argmax(l):
	return l.index(max(l))

def harmonic_mean(x, y):
	if x == y == 0:
		return 0
	return (2 * x * y) / float(x + y)


def levenstein(i, j, s, t):
	return 0 if s[i] == t[j] else 1


def close_match(str1, str2, threshold):
	if str1 == str2:
		return True
	norm = float(len(str1) + len(str2))
	min_dist = abs(len(str1) - len(str2)) / norm 
	if min_dist < threshold:
		dist = edit_distance(str1, str2, 1, levenstein)
		return ((dist <= 1) or (dist / norm) < threshold)
	return False


# Operations include skip or match
# Match cost is dependent on the two extrema
def edit_distance(s, t, id_cost, match_f):
	'''
	:param s: 0 sequence 1
	:param t: 0 sequence 2
	:id_cost: num Cost of an Insertion or Deletion operation
	:match_f: func (idx1, idx2, s, t) -> num  Cost of matching
	:return: Edit distance between s and t
	'''
	l1 = len(s) + 1 # width
	l2 = len(t) + 1 # height
	d = [ [x * id_cost for x in xrange(l2)] ]

	for i in xrange(1, l1):
		d.append([i * id_cost])
		for j in xrange(1, l2):
			_del = d[i-1][j] + id_cost
			_ins = d[i][j-1] + id_cost
			_match = match_f(i-1, j-1, s, t) + d[i-1][j-1]
			d[i].append(min(_del, _ins, _match))
	i = l1 - 1
	j = l2 - 1
	final_val =  d[l1 - 1][l2 - 1] 
	return final_val


def get_docs(data_dir, pr=True):
	docs = []
	num_loaded = 0
	num_exceptions = 0
	for _dir in os.listdir(data_dir):
		print "Starting Dir: ", _dir
		r_dir = os.path.join(data_dir, _dir)
		basenames = set()
		image_names = os.listdir(r_dir)
		for name in image_names:
			if name.endswith(".jpg"):
				basename = os.path.splitext(name)[0]
				basenames.add(basename)
		for basename in basenames:
			#try:
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
			document = doc.Document(basename, image_path, ocr_path, prof_path, form_path)
			docs.append(document)
			num_loaded += 1
			if pr and num_loaded % 10 == 0:
				print "Loaded %d documents" % num_loaded
			#except:
			#	if pr:
			#		print "Exception reading ", os.path.join(_dir, basename)
			#	num_exceptions += 1
	if pr:
		print "%d Docs read" % num_loaded
		print "%d Docs could not be read" % num_exceptions
	return docs

