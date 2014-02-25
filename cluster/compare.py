
import math
import components

_debug = False

def dist(p1, p2):
	return math.sqrt( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 )


def has_match(line, search_lines, thresh_dist):

	if line.matched:
		return True
	
	for cmp_line in search_lines:
		if cmp_line.matched or dist(line.pos, cmp_line.pos) > thresh_dist:
			# optimization
			continue

		if close_match(line.text, cmp_line.text):
			# yes it is a match
			line.matched = True
			cmp_line.matched = True
			if _debug:
				print line
			return True
		elif line.text.startswith(cmp_line.text + " ") or line.text.endswith(" " + cmp_line.text):
			# + " " makes sure that we are matching complete words
			if line.text.startswith(cmp_line.text + " "):
				chars = line.chars[cmp_line.N:]
			else:
				chars = line.chars[0:cmp_line.N]
			first_char = chars[0]
			last_char = chars[-1]
			pos1 = first_char.pos
			pos2 = last_char.pos
			size = (pos2[0] - pos1[0], line.size[1])

			# calculating the position of the substring is approximated, but could be exact if we get it from the OCR
			remaining_line = components.TextLine(chars, pos1, size)
			remaining_line.matched = False
			if _debug:
				print "partial match:"
				print "\t", cmp_line
				print "\t", line
				print "\t", "Searching for remaining:", remaining_line

			tmp = has_match(remaining_line, search_lines, thresh_dist)
			if tmp:
				line.matched = True
				cmp_line.matched = True
				return True
	return False
				

def compute_matching(doc1, doc2):
	lines1 = doc1.textLines
	lines2 = doc2.textLines
	thresh_dist = 0.10 * max(max(doc1.size), max(doc2.size))  # % of larger dimension

	for line in lines1:
		line.matched = False
	for line in lines2:
		line.matched = False

	# each matched line has a flag set by has_match indicating that it matches
	# we do it both ways for substring matching.
	map(lambda line: has_match(line, lines2, thresh_dist), lines1)
	map(lambda line: has_match(line, lines1, thresh_dist), lines2)
	filtered1 = filter(lambda line: line.matched, lines1)
	filtered2 = filter(lambda line: line.matched, lines2)

	# could replace with reduce...
	chars1 = sum(map(lambda line: len(line.text), filtered1))
	chars2 = sum(map(lambda line: len(line.text), filtered2))
	total_chars = doc1.charCount + doc2.charCount

	return (chars1 + chars2) / float(total_chars)
	

def close_match(str1, str2, threshold=0.20):
	if str1 == str2:
		return True
	norm = float(len(str1) + len(str2))
	min_dist = abs(len(str1) - len(str2)) / norm 
	if min_dist < threshold:
		dist = edit_distance(str1, str2, 1,  lambda i, j, s, t: 0 if s[i] == t[j] else 1)
		return ((dist <= 1) or (dist / norm) < threshold)
	return False
	
	
def compare(doc1, doc2):
	size1 = doc1.size
	size2 = doc2.size
	percent_matching = compute_matching(doc1, doc2)
	return percent_matching

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


if __name__ == "__main__":
	_str = "abc"
	for other in ['abc', 'abcd', 'bbc', 'qbc', 'akdsjd', 'abcdefgh']:
		print _str, other, close_match(_str, other)
	
