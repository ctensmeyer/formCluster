
import os
import math
import string
import Levenshtein

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
			 (255, 255, 0), (255, 0, 255), (0, 255, 255),
			 (128, 0, 0), (0, 128, 0), (0, 0, 128),
			 (128, 128, 0), (128, 0, 128), (0, 128, 128)]

def e_dist(p1, p2):
	return math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 )

def argmax(l):
	return l.index(max(l))

def harmonic_mean(x, y, beta=1.0):
	'''
	beta > 1 biases mean toward y
	beta < 1 biases mean toward x
	'''
	if x == y == 0:
		return 0
	return ((1 + beta) * x * y) / float((beta * x) + y)

def harmonic_mean_list(l):
	'''
	:param l: list of nums
	'''
	if not l:
		return 0.0
	prod = float(reduce(lambda x, y: x * y, l))
	if prod == 0:
		return prod
	denum = sum(map(lambda x: prod / x, l))
	return len(l) * prod / denum


def avg(l):
	return float(sum(l)) / len(l) if len(l) else float('nan')


def stddev(l):
	mean = avg(l)
	var = sum(map(lambda x: (x - mean) ** 2, l)) / len(l)
	return math.sqrt(var)
	

def levenstein(i, j, s, t):
	return 0 if s[i] == t[j] else 1


def close_match(str1, str2, threshold):
	if str1 == str2:
		return True
	norm = float(len(str1) + len(str2))
	min_dist = abs(len(str1) - len(str2)) / norm 
	if min_dist < threshold:
		dist = edit_distance(str1, str2, 1, levenstein)
		#dist = Levenshtein.distance(str1, str2)
		return ((dist <= 1) or (dist / norm) < threshold)
	return False


def apply_mat(mat, func):
	new_mat = []
	for row in mat:
		new_mat.append(map(func, row))
	return new_mat


def format_as_mat(mat):
	'''
	:param mat: {<set x> : {<set y> : <any> } }
	'''
	new_mat = []
	for x in sorted(mat.keys()):
		row = []
		for y in sorted(mat[x].keys()):
			row.append(mat[x][y])
		new_mat.append(row)
	return new_mat


def pairwise(args, func, symmetric=True):
	mat = []
	for x, arg1 in enumerate(args):
		row = []
		for y, arg2 in enumerate(args):
			if symmetric and y < x:
				val = mat[y][x]
			else:
				val = func(arg1, arg2)
			row.append(val)
		mat.append(row)
	return mat

def insert_indices(mat, row_start=0, col_start=0):
	row0 = range(col_start, len(mat[0]) + col_start)
	row0.insert(0, " ")
	for x,row in enumerate(mat, row_start):
		row.insert(0, x)
	mat.insert(0, row0)

def print_mat(mat):
	max_lens = [max([len(str(r[i])) for r in mat])
					 for i in range(len(mat[0]))]

	print "\n".join(["".join([string.rjust(str(e), l + 2)
							for e, l in zip(r, max_lens)]) for r in mat])

def split_mat(mat, row_len):
	mats = []
	total_row_length = len(mat[0])
	start = 0
	end = row_len
	while start < total_row_length:
		new_mat = []
		for row in mat:
			new_row = row[start:end]
			new_mat.append(new_row)
		mats.append(new_mat)
		start += row_len
		end += row_len
	return mats



	

# Operations include skip or match
def edit_distance(s, t, id_cost, match_f):
	'''
	:param s: sequence 1
	:param t: sequence 2
	:param id_cost: num Cost of an Insertion or Deletion operation
	:param match_f: func (idx1, idx2, s, t) -> num  Cost of matching
	:return: Edit distance between s and t
	'''
	l1 = len(s) + 1 # height
	l2 = len(t) + 1 # width
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

if __name__ == "__main__":
	mat = pairwise(xrange(5), lambda x,y: math.sqrt(x + y))
	insert_indices(mat, row_start=2, col_start=3)
	print_mat(mat)

