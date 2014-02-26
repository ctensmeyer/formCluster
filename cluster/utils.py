
import os
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


