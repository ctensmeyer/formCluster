
from kmeans import KMeans
import collections
import math
import ast
import json
import components

_true_offset_file = '/home/chris/Ancestry/Data/1911England/images/offsets.txt'
_true_offsets = None


def get_true_offsets():
	d = collections.defaultdict(dict)
	_in = open(_true_offset_file, 'r')
	for line in _in.readlines():
		tokens = line.split()
		if len(tokens):
			one = int(tokens[0])
			two = int(tokens[1])
			offset = ast.literal_eval(tokens[2] + tokens[3])
			d[one][two] = offset
	return d

_true_offsets = get_true_offsets()

def compute_diffs(d1, d2, scale):
	'''
	:param d1: dict of ele to list( (x, y) )
	:param d2: dict of ele to list( (x, y) )
	:param scale: (x, y) scale coordinates of d2 to coordinates of d1
	Note that d1/2 should be default dicts to [], so we don't have to worry about
		having the exact same domains.
	Diffs are the offset of d2 with respect to d1
	'''
	diffs = []
	for ele in set(d1.keys() + d2.keys()):
		locs1 = d1[ele]
		locs2 = d2[ele]
		for loc1 in locs1:
			for loc2 in locs2:
				loc2 = (loc2[0] * scale[0], loc2[1] * scale[1])
				diff = (loc2[0] - loc1[0], loc2[1] - loc1[1])
				diffs.append(diff)
	return diffs

def form_text_line_list(lines):
	d = collections.defaultdict(list)
	for line in lines:
		d[line.text].append(line.position)
	return d
	

def compute_offset(doc1, doc2, scale):
	return _true_offsets[doc1._id][doc2._id]

	#list1 = form_text_line_list(doc1.textLines)
	#list2 = form_text_line_list(doc2.textLines)
	#print "\tTotal number of Text Lines: %d & %d" % (len(list1), len(list2))
	#diffs = compute_diffs(list1, list2, scale)
	#print "\tNumber of Text Line Pairs in common: %d" % len(diffs)
	##with open('data.plot', 'w') as f:
	##	for diff in diffs:
	##		f.write("%d %d\n" % diff)
	#if diffs:
	#	kmeans = KMeans(diffs, 2)
	#	kmeans.cluster()
	#	#kmeans.display()
	#	return kmeans.largest_cluster()  # change this later

def transform(p, scale, offset):
	return (p[0] * scale[0] - offset[0], p[1] * scale[1] - offset[1])
	
def dist(p1, p2):
	return math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 )


def has_match(line, search_lines, scale, offset, acceptable_distance):

	if line.matched:
		return True
	
	for cmp_line in search_lines:
		cmp_pos = transform(cmp_line.position, scale, offset)

		if cmp_line.matched or dist(line.position, cmp_pos) > acceptable_distance:
			# optimization
			continue

		if close_match(line.text, cmp_line.text):
			# yes it is a match
			line.matched = True
			cmp_line.matched = True
			print line
			return True
		elif line.text.startswith(cmp_line.text + " ") or line.text.endswith(" " + cmp_line.text):
			# + " " makes sure that we are matching complete words
			remaining_text = line.text.replace(cmp_line.text, "").strip()

			# calculating the position of the substring is approximated, but could be exact if we get it from the OCR
			idx = line.text.find(remaining_text)
			remaining_pos = (line.position[0] + (idx / float(len(line.text))) * line.size[0], line.position[1]) 
			remaining_size = ((idx / float(len(line.text))) * line.size[0], line.size[1]) 
			remaining_line = components.TextLine(remaining_text, line.confidences[idx:], remaining_pos, remaining_size)
			remaining_line.matched = False
			print "partial match:"
			print "\t", cmp_line
			print "\t", line
			print "\t", "Searching for remaining:", remaining_line

			tmp = has_match(remaining_line, search_lines, scale, offset, acceptable_distance)
			if tmp:
				line.matched = True
				cmp_line.matched = True
				return True
	return False
				

def compute_matching2(doc1, doc2, scale, offset):
	lines1 = doc1.textLines
	lines2 = doc2.textLines
	acceptable_distance = 0.10 * max(doc1.size)  # % of larger dimension

	for line in lines1:
		line.matched = False
	for line in lines2:
		line.matched = False

	# each matched line has a flag set by has_match indicating that it matches
	# we do it both ways for substring matching.
	map(lambda line: has_match(line, lines2, scale, offset, acceptable_distance), lines1)
	map(lambda line: has_match(line, lines1, (1.0 / scale[0], 1.0 / scale[1]), (-offset[0], -offset[1]), acceptable_distance), lines2)
	filtered1 = filter(lambda line: line.matched, lines1)
	filtered2 = filter(lambda line: line.matched, lines2)

	chars1 = sum(map(lambda line: len(line.text), filtered1))
	chars2 = sum(map(lambda line: len(line.text), filtered2))
	total_chars = sum(map(lambda line: len(line.text), lines1) + map(lambda line: len(line.text), lines2))  

	return (chars1 + chars2) / float(total_chars)
	

def compute_matching(doc1, doc2, scale, offset):
	# higher return is better
	lines1 = doc1.textLines
	lines2 = doc2.textLines
	total_text_lines = len(lines1) + len(lines2)
	matching_text_lines = 0
	acceptable_distance = 0.10 * max(doc1.size)  # % of larger dimension

	for line1 in lines1:
		for line2 in lines2:
			#if line1.text == line2.text:

			pos1 = line1.position
			pos2 = transform(line2.position, scale, offset)
			if dist(pos1, pos2) <= acceptable_distance:
				if close_match(line1.text, line2.text):
					print repr(line1.text)
					matching_text_lines += 1
				elif line2.text.startswith(line1.text) or line2.text.endswith(line1.text):
					remaining = line2.text.replace(line1.text, "").strip()
					idx = line2.text.find(remaining)
					remaining_pos = (pos2[0] + (idx / float(len(line2.text))) * line2.size[0], pos2[1])
					print "Substring match: %r is in %r" % (line1.text, line2.text)
					print "Looking for %r" % remaining
					# search through lines1
					for _line1 in lines1:
						#if remaining == _line1.text:
						_pos1 = _line1.position
						if dist(remaining_pos, _pos1) < acceptable_distance and close_match(remaining, _line1.text):
							print "Found Remaining : ", repr(remaining)
							matching_text_lines += 1

				elif  line1.text.startswith(line2.text) or line1.text.endswith(line2.text):
					remaining = line1.text.replace(line2.text, "").strip()
					idx = line1.text.find(remaining)
					remaining_pos = (pos1[0] + (idx / float(len(line1.text))) * line1.size[0], pos1[1])
					print "Substring match: %r is in %r" % (line2.text, line1.text)
					print "Looking for %r" % remaining
					# search through lines1
					for _line2 in lines2:
						#if remaining == _line2.text:
						_pos2 = _line2.position
						if dist(remaining_pos, _pos2) < acceptable_distance and close_match(remaining, _line2.text):
							print "Found Remaining : ", repr(remaining)
							matching_text_lines += 1
				
	print "Total Matches:", matching_text_lines
	return matching_text_lines / float(total_text_lines)

def close_match(str1, str2, threshold=0.10):
	if str1 == str2:
		return True
	min_dist = abs(len(str1) - len(str2)) / (len(str1) + len(str2))
	if min_dist < threshold:
		return _edit_distance(str1, str2, 1,  lambda i, j, s, t: 0 if s[i] == t[j] else 1) < threshold
	return False
	
	
# we scale the coordinates of doc2 to match doc1
def compare(doc1, doc2):
	size1 = doc1.size
	size2 = doc2.size
	scale = ( size1[0] / float(size2[0]), size1[1] / float(size2[1]))
	offset = compute_offset(doc1, doc2, scale)
	print "\tScale: %s\tOffset: %s" % (scale, offset)
	if offset:
		percent_matching = compute_matching2(doc1, doc2, scale, offset)
		return percent_matching
	else:
		return 0.0

# Operations include skip or match
# Match cost is dependent on the two extrema
def _edit_distance(s, t, id_cost, match_f):
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
	norm = final_val / float(len(s) + len(t))
	return norm
	


if __name__ == "__main__":
	_str = "abc"
	for other in ['abc', 'abcd', 'bbc', 'qbc', 'akdsjd', 'abcdefgh']:
		print _str, other, close_match(_str, other)
	
	#print json.dumps(get_true_offsets(), indent=4)
