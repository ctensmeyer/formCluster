
import Image
import ImageDraw
import xml.etree.ElementTree as ET
import math

import components
import utils

def _read_line(ele, orien):
	'''
	local function to process a single xml line element node
	'''
	x = int(ele.get('x'))
	y = int(ele.get('y'))
	l = int(ele.get("length"))
	t = int(ele.get("thickness"))
	line = components.Line(orien, (x, y), l, t)
	return line

def sort_lines(lines):
	lines.sort(key=lambda line: (line.pos[line.orien], line.pos[1 - line.orien]))
	

def read_lines(path):
	'''
	Reads in lines from an xml file specified by path
	'''
	tree = ET.parse(path)
	root = tree.getroot()
	h_lines = list()
	v_lines = list()
	for line_ele in root.find('HorizontalLines'):
		h_lines.append(_read_line(line_ele, components.Line.HORIZONTAL))
	for line_ele in root.find('VerticalLines'):
		v_lines.append(_read_line(line_ele, components.Line.VERTICAL))
	sort_lines(h_lines)
	sort_lines(v_lines)
	line_up_colinears(h_lines)
	line_up_colinears(v_lines)
	sort_lines(h_lines)
	sort_lines(v_lines)
	return h_lines, v_lines


def _overlap(a, b, x, y):
	'''
	Returns True if the range (a, b) overlaps the range (x, y)
	'''
	assert a < b
	assert x < y
	# make sure that a is the smaller endpoint
	if a > x:
		a, b, x, y = x, y, a, b
	return b > x 
	

def line_up_colinears(lines):
	'''
	Makes lines that are approximately colinear, colinear
	:param lines: list of lines
	'''
	#TODO: this only handles the case of two colinear lines.  It could occur with more
	if not lines:
		return
	o = lines[0].orien
	_range = lines[-1].pos[o] - lines[0].pos[o]
	thresh = _range / 500.0 + 5  # TODO: play with this.  depends on skew estimate
	for idx in xrange(len(lines) - 1):
		cur_line = lines[idx]
		next_line = lines[idx + 1]
		diff = next_line.pos[o] - cur_line.pos[o] 
		if diff < thresh:
			# check that they don't overlap
			a, b = next_line.pos[1-o], next_line.pos[1-o] + next_line.length
			x, y = cur_line.pos[1-o], cur_line.pos[1-o] + cur_line.length
			if not _overlap(a, b, x, y):
				# average the positions
				#print "Colinear"
				#print cur_line
				#print next_line
				val = (next_line.pos[o] + cur_line.pos[o]) / 2
				tmp = list(next_line.pos)
				tmp[o] = val
				next_line.pos = tuple(tmp)
				tmp = list(cur_line.pos)
				tmp[o] = val
				cur_line.pos = tuple(tmp)

def draw_lines(h, v, image_path, size):
	im = Image.new('RGB', size, 'white')
	draw = ImageDraw.Draw(im)
	for line in h:
		draw.line( (line.pos, (line.pos[0] + line.length, line.pos[1]) ) , width=(line.thickness * 2),
					fill=('purple' if line.matched else 'red'))
	for line in v:
		draw.line( (line.pos, (line.pos[0],  line.pos[1] + line.length) ) , width=(line.thickness * 2),
					fill=('orange' if line.matched else 'blue'))
	im.save(image_path)


class LineSequenceMatcher():
	# constants used in our back pointer table for Dynamic Programming
	START = -1
	MATCH = 0
	DEL_1 = 1
	DEL_2 = 2
	SUB = 3

	def __init__(self, lines1, lines2, dist_thresh):
		'''
		:param lines1: list of Lines
		:param lines2: list of Lines
		'''
		# makes sure that we only have one orientation between both lists
		assert len(set(map(lambda line: line.orien, lines1 + lines2))) == 1
		self.lines1 = lines1
		self.lines2 = lines2
		self.dist_thresh = dist_thresh

	def sort(self):
		'''
		Sorts the sequence of lines
		TODO: handle cases where lines are really close together
		'''
		self.lines1.sort(key=lambda line: line.pos[line.orien])
		self.lines2.sort(key=lambda line: line.pos[line.orien])

	def _get_last_match_indices(self, i, j):
		'''
		Backtraces op_mat to find the indices of the last match (if any)
			given start indices i,j
		'''
		s = LineSequenceMatcher.START
		m = LineSequenceMatcher.MATCH
		d1 = LineSequenceMatcher.DEL_1
		d2 = LineSequenceMatcher.DEL_2
		while True:
			o = self.op_mat[i][j]
			if o == s or o == m:
				break
			if o == d1:
				i -= 1
			elif o == d2:
				j -= 1
			else:
				# substitution
				i -= 1
				j -= 1
		return i - 1, j - 1  # -1 to convert from table indicies to list indices

	def line_indel_cost(self, line):
		return 2

	def line_match_cost(self, l1_ci, l2_ci, l1_pi, l2_pi):
		'''
		:param l1_ci: int list 1 current index - index into list 1
		:param l2_ci: int list 2 current index - index into list 2
		:param l1_pi: int list 1 previous index - index into list 1 for the previous matched line
		:param l2_pi: int list 2 previous index - index into list 2 for the previous matched line
		'''
		line1 = self.lines1[l1_ci]
		line2 = self.lines2[l2_ci]
		o = line1.orien

		# Ignore thickness for now
		#dt = abs(line1.thickness - line2.thickness)
		len_ratio = max(line1.length / float(line2.length), line2.length / float(line1.length))
		assert len_ratio >= 1.0
		
		if l1_pi == -1 and l2_pi == -1:
			# matching first line
			dist = 0
		else:
			dist1 = line1.pos[o] - self.lines1[l1_pi].pos[o]
			dist2 = line2.pos[o] - self.lines2[l2_pi].pos[o]
			assert dist1 >= 0
			assert dist2 >= 0
			diff_dist = dist1 - dist2
			
			offset1 = line1.pos[1-o] - self.lines1[l1_pi].pos[1-o]
			offset2 = line2.pos[1-o] - self.lines2[l2_pi].pos[1-o]
			diff_offset = offset1 - offset2;
			dist = math.sqrt(diff_dist ** 2 + diff_offset ** 2)

		if len_ratio < 1.15 and dist < self.dist_thresh:
			return 0
		else:
			return 3
	
		
	def _init_tables(self):
		'''
		Initializes the cost_matrix and the operations_matrix (back pointers) '''
		# seed the first row with cumulative costs of deleting lines2
		#  as if lines1 were considered the empty sequence
		first_row = [0]
		for j in xrange(1, len(self.lines2) + 1):
			val = self.line_indel_cost(self.lines2[j-1]) + first_row[-1]
			first_row.append(val)
		self.cost_mat = [first_row]

		# create the operations mat of essentially, back pointers
		self.op_mat = []
		op_first_row = [LineSequenceMatcher.DEL_2 for x in xrange(len(self.lines2) + 1)]
		op_first_row[0] = LineSequenceMatcher.START
		self.op_mat.append(op_first_row)

	def _init_new_row(self, i):
		'''
		Initializes a new row for another pass in line_edit_distance()
		'''
		# start each successive row with the cost as if lines2 were
		#  an empty sequence, so the cumulative cost of deleting lines1
		val = self.line_indel_cost(self.lines1[i-1]) + self.cost_mat[-1][0]
		self.cost_mat.append([val])
		self.op_mat.append([LineSequenceMatcher.DEL_1])
		
	def _calc_costs(self, i, j):
		'''
		Calculates the cumulative costs of deleting from either sequence or matching
			the currently considered line pair at table indices i,j
		'''
		# cost of deleting current line from lines1
		tmp1 = (self.lines1[i-1])
		del_line1_cost = self.cost_mat[i][j-1] + self.line_indel_cost(tmp1)
		# cost of deleting current line from lines2
		tmp2 = (self.lines2[j-1])
		del_line2_cost = self.cost_mat[i-1][j] + self.line_indel_cost(tmp2)

		# match cost depends on the relative distance of each of the two
		#  lines in consideration and the previous set of lines set as a match
		lines1_idx, lines2_idx = self._get_last_match_indices(i-1, j-1)

		marginal_match_cost = self.line_match_cost(i-1, j-1, lines1_idx, lines2_idx)
		cum_match_cost = marginal_match_cost + self.cost_mat[i-1][j-1]
		return del_line1_cost, del_line2_cost, cum_match_cost, marginal_match_cost

	def _update_table(self, dl1, dl2, mc, match):
		'''
		:param dl1: cumulative cost of deleting from lines1
		:param dl2: cumulative cost of deleting from lines2
		:param mc: cumulative cost of matching lines
		:param match: boolean match for True, substitution for False
		'''
		_min = min(dl1, dl2, mc)
		self.cost_mat[-1].append(_min)
		if _min == dl1:
			self.op_mat[-1].append(LineSequenceMatcher.DEL_2)
		elif _min == dl2:
			self.op_mat[-1].append(LineSequenceMatcher.DEL_1)
		elif _min == mc:
			self.op_mat[-1].append(LineSequenceMatcher.MATCH if match else LineSequenceMatcher.SUB)
		else:
			assert False  # ???

	def line_edit_distance(self):
		'''
		This is a heuristic to match two sequences of line objects based on the
			principles of the edit distance.
		This does not always return the "optimal" solution because the solution to
			a problem *does* depend on the choice of operations made in sub problems.
			This dependency occurs because the cost of matching a pair of lines depends on
			having a common point of reference (ie, previous matching lines) to calculate the
			difference in their distances to the reference point.
		'''
		self._init_tables()
		for i in xrange(1, len(self.lines1) + 1):
			self._init_new_row(i)
			for j in xrange(1, len(self.lines2) + 1):
				del_line1_cost, del_line2_cost, cum_match_cost, marginal_match_cost = self._calc_costs(i, j)
				self._update_table(del_line1_cost, del_line2_cost, cum_match_cost, marginal_match_cost == 0)
		final_val =  self.cost_mat[-1][-1] 
		return final_val

	def mark_matches(self):
		'''
		Assumes self.op_mat is built
		Backtraces the entire sequence and marks every pair of lines that match
		'''
		i = len(self.op_mat) - 1
		j = len(self.op_mat[0]) - 1
		while self.op_mat[i][j] !=  LineSequenceMatcher.START:
			if self.op_mat[i][j] == LineSequenceMatcher.DEL_1:
				i -= 1
			elif self.op_mat[i][j] == LineSequenceMatcher.DEL_2:
				j -= 1
			elif self.op_mat[i][j] == LineSequenceMatcher.SUB:
				i -= 1
				j -= 1
			elif self.op_mat[i][j] == LineSequenceMatcher.MATCH:
				i -= 1
				j -= 1
				self.lines1[i].matched = True
				self.lines2[j].matched = True
			else:
				assert False

	def get_matches(self):
		'''
		Assumes self.op_mat is built
		Backtraces the entire sequence and stores every pair of matching lines
		'''
		lines = list()
		i = len(self.op_mat) - 1
		j = len(self.op_mat[0]) - 1
		while self.op_mat[i][j] !=  LineSequenceMatcher.START:
			if self.op_mat[i][j] == LineSequenceMatcher.DEL_1:
				i -= 1
			elif self.op_mat[i][j] == LineSequenceMatcher.DEL_2:
				j -= 1
			elif self.op_mat[i][j] == LineSequenceMatcher.SUB:
				i -= 1
				j -= 1
			elif self.op_mat[i][j] == LineSequenceMatcher.MATCH:
				i -= 1
				j -= 1
				lines.append( (self.lines1[i], self.lines2[j]) )
			else:
				assert False
		return lines
	
	def display(self):
		'''
		prints the cost_mat and the op_mat
		'''
		print "Cost Matrix"
		utils.print_mat(self.cost_mat)
		print "Operations Matrix"
		utils.print_mat(self.op_mat)


