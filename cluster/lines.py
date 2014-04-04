
import xml.etree.ElementTree as ET
import ImageDraw
import Image
import munkres
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
			if not utils.are_overlapping([cur_line.length_range(), next_line.length_range()]):
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


class LineMatcher():
	
	def __init__(self, lines1, lines2):
		'''
		:param lines1: list of Lines
		:param lines2: list of Lines
		'''
		# makes sure that we only have one orientation between both lists
		assert len(set(map(lambda line: line.orien, lines1 + lines2))) <= 1
		self.lines1 = lines1
		self.lines2 = lines2

	def get_matches(self):
		'''
		Performs Line matching.
		:return: list( (line1, line2) )
		'''
		return zip(self.lines1, self.lines2)

	def sort(self):
		'''
		Sorts the sequence of lines
		TODO: handle cases where lines are really close together
		'''
		self.lines1.sort(key=lambda line: line.pos[line.orien])
		self.lines2.sort(key=lambda line: line.pos[line.orien])


class LMatcher(LineMatcher):
	'''
	Uses dyanmic programming like edit distance with operations specific to lines.
	After the edit distance is computed, the algorithm looks for transposes.
	The main idea is to slightly greedily detect the perfect matches, which serve
	as reference points for dealing with the coordinate offset problem.  Then we
	fine tune and match as many things as possible.  Also this class allows for
	line sequence merging based on partial matches.
	'''
	EMPTY = -2               # initialized value
	START = -1               # value at the start corner of the table
	PERFECT = 0              # both lines are a perfect match 
	DEL1 = 1                 # delete line1
	DEL2 = 2                 # delete line2
	CONTAINS1 = 3            # line1 contains line2
	CONTAINS2 = 4            # line2 contains line1
	OVERLAP = 5              # the lines overlap each other
	CONNECT1 = 6             # line1 overlaps two lines in the other sequence
	CONNECT2 = 7             # line2 overlaps two lines in the other sequence
	COFRAG = 8               # line1 and line2 are probable fragments of the same larger line
	NO_MATCH = 9             # indicates that line1 and line2 do not match any of the above cases
	NO_MATCH_COST = 1e10     # Arbitrary high cost to cause certain operations to be never chosen as optimal

	OP_STR = {EMPTY: "Empty", START: "Start", PERFECT: "Perfect", CONTAINS1: "Contains1",
				 CONTAINS2: "Contains2", OVERLAP: "Overlap", CONNECT1: "Connect1", CONNECT2: "Connect2",
				 COFRAG: "Cofrag", DEL1: "Del1", DEL2: "Del2", NO_MATCH: "No_Match"}

	def __init__(self, lines1, lines2, dist_thresh):
		LineMatcher.__init__(self, lines1, lines2)
		self.dist_thresh = dist_thresh
		self.offset_thresh = dist_thresh / 2
		self.tables_built = False

	def _table_check(self):
		if not self.tables_built:
			self.build_tables()
		self.tables_built = True

	def op_str(self, op):
		return self.OP_STR.get(op)

	def indel_cost(self, line):
		'''
		Cost of deleting/inserting a line.  Linear to the number of times a line occurs.
		Biases algorithm to match lines with a large count.
		'''
		return line.length * line.count

	def contains_cost(self, container, containee):
		'''
		Cost of line1 containing line2
		Biases algorithm to match lines with a large count.
		:param container: containing line
		:param containee: contained line
		'''
		return (container.length - containee.length) * (container.count + containee.count) / 2

	def connect_cost(self, connector, line1, line2):
		'''
		Cost of connector connecting line1 and line2
		Assumes the lines are approx colinear
		:param connector: line connecting line1/2
		:param line1/2: line
		'''
		o = connector.orien
		g_len = utils.gap_len(line1.length_range(), line2.length_range())
		cost =  g_len * (connector.count + ( (line1.count + line2.count) / 2)) # double count the gap
		if (line1.pos[1-o] > line2.pos[1-o]):
			line1, line2 = line2, line1

		# count the cost if line1 and line2 don't align perfectly with the ends of connector
		start_diff = abs(connector.length_range()[0] - line1.length_range()[0])
		cost += start_diff * ( (connector.count + line1.count) / 2.0)
		end_diff = connector.length_range()[1] - line2.length_range()[1]
		cost += end_diff * ( (connector.count + line2.count) / 2.0)
		return cost

	def cofrag_cost(self, line1, line2):
		'''
		Cost of saying line1 and line2 are cofragments
		Bias is toward this operation over two successive indels
		'''
		return (self.indel_cost(line1) + self.indel_cost(line2) - 1)

	def overlap_cost(self, line1, line2):
		'''
		Cost of saying line1 and line2 overlap
		'''
		avg_count = (line1.count + line2.count) / 2.0 
		o_len = utils.overlap_len(line1.length_range(), line2.length_range())
		return avg_count * (line1.length + line2.length - 2 * o_len)

	def match_cost_type(self, i, j):
		line1 = self.lines1[i-1]
		line2 = self.lines2[j-1]
		o = line1.orien
		prev_line_idxs = self.last_match_mat[i-1][j-1]

		# calculate distance and offset
		if prev_line_idxs:
			prev1 = self.lines1[prev_line_idxs[0] - 1]
			prev2 = self.lines2[prev_line_idxs[1] - 1]

			dist1 = line1.pos[o] - prev1.pos[o]
			dist2 = line2.pos[o] - prev2.pos[o]
			assert dist1 >= 0
			assert dist2 >= 0
			dist = abs(dist1 - dist2)
			offset1 = line1.pos[1-o] - prev1.pos[1-o]
			offset2 = line2.pos[1-o] - prev2.pos[1-o]
			offset = abs(offset1 - offset2)
		else:
			# we could do a small penalty term for large divergences
			dist = 0
			offset = 0
		len_ratio = max(line1.length / float(line2.length), line2.length / float(line1.length))

		# anything not the proper distance away is not considered
		if dist > self.dist_thresh:
			return self.NO_MATCH_COST, self.NO_MATCH
		elif offset < self.offset_thresh and len_ratio < 1.15:
			return 0, self.PERFECT
		elif utils.range_contains(line1.length_range(), line2.length_range()):
			return self.contains_cost(line2, line1), self.CONTAINS2
		elif utils.range_contains(line2.length_range(), line1.length_range()):
			return self.contains_cost(line1, line2), self.CONTAINS1
		elif utils.are_overlapping([line1.length_range(), line2.length_range()]):
			return self.overlap_cost(line1, line2), self.OVERLAP
		else:
			# in the case of similarly spaced lines, slight bias toward COFRAG over DEL1/DEL2 pair
			return self.cofrag_cost(line1, line2), self.COFRAG

	def connect1_cost(self, i, j):
		'''
		Assumes line i and line j are labeled as overlapping by self.match_cost_type(i, j)
		:return: marginal cost of assuming that line i connects line j and line j-1
		'''
		if j == 1:
			return self.NO_MATCH_COST, self.NO_MATCH
		line1 = self.lines1[i-1]
		line21 = self.lines2[j-1]
		line22 = self.lines2[j-2]

		r1 = line1.length_range()
		r21 = line21.length_range()
		r22 = line22.length_range()
		if self.are_colinear(line21, line22) and utils.are_overlapping([r1, r21, r22]):
			return self.connect_cost(line1, line21, line22), self.CONNECT1
		else:
			return self.NO_MATCH_COST, self.NO_MATCH

	def connect2_cost(self, i, j):
		'''
		Assumes line i and line j are labeled as overlapping by self.match_cost_type(i, j)
		:return: marginal cost of assuming that line j connects line i and line i-1
		'''
		if i == 1:
			return self.NO_MATCH_COST, self.NO_MATCH
		line2 = self.lines2[j-1]
		line11 = self.lines1[i-1]
		line12 = self.lines1[i-2]

		ranges = map(lambda line: line.length_range(), [line2, line11, line12])
		if self.are_colinear(line11, line12) and utils.are_overlapping(ranges):
			return self.connect_cost(line2, line11, line12), self.CONNECT2
		else:
			return self.NO_MATCH_COST, self.NO_MATCH
		
	def build_tables(self):
		self.init_tables()
		for i in xrange(1, len(self.lines1) + 1):
			self.op_mat[i][0] = self.DEL1
			self.cost_mat[i][0] = self.cost_mat[i-1][0] + self.indel_cost(self.lines1[i-1])
			for j in xrange(1, len(self.lines2) + 1):
				# marginal costs
				del1_marg_cost = self.indel_cost(self.lines1[i-1])
				del2_marg_cost = self.indel_cost(self.lines2[j-1])
				match_marg_cost, match_type = self.match_cost_type(i, j)
				conn1_marg_cost, conn1_type = self.connect1_cost(i, j) if match_type in [self.OVERLAP, self.CONTAINS1] else (self.NO_MATCH_COST, self.NO_MATCH)
				conn2_marg_cost, conn2_type = self.connect2_cost(i, j) if match_type in [self.OVERLAP, self.CONTAINS2] else (self.NO_MATCH_COST, self.NO_MATCH)

				# cumulative costs
				del1_cum_cost = del1_marg_cost + self.cost_mat[i-1][j]
				del2_cum_cost = del2_marg_cost + self.cost_mat[i][j-1]
				match_cum_cost = match_marg_cost + self.cost_mat[i-1][j-1]
				conn1_cum_cost = conn1_marg_cost + (self.cost_mat[i-1][j-2] if j > 1 else 1e10)
				conn2_cum_cost = conn2_marg_cost + (self.cost_mat[i-2][j-1] if i > 1 else 1e10)

				# update tables
				_min = min(del1_cum_cost, del2_cum_cost, match_cum_cost, conn1_cum_cost, conn2_cum_cost)
				if _min == del1_cum_cost:
					self.cost_mat[i][j] = del1_cum_cost
					self.op_mat[i][j] = self.DEL1
					self.last_match_mat[i][j] = self.last_match_mat[i-1][j]

				elif _min == del2_cum_cost:
					self.cost_mat[i][j] = del2_cum_cost
					self.op_mat[i][j] = self.DEL2
					self.last_match_mat[i][j] = self.last_match_mat[i][j-1]

				elif _min == match_cum_cost:
					self.cost_mat[i][j] = match_cum_cost
					self.op_mat[i][j] = match_type
					if match_type == self.PERFECT:
						self.last_match_mat[i][j] = (i, j)
					else:
						self.last_match_mat[i][j] = self.last_match_mat[i-1][j-1]

				elif _min == conn1_cum_cost:
					self.cost_mat[i][j] = conn1_cum_cost
					self.op_mat[i][j] = self.CONNECT1
					self.last_match_mat[i][j] = self.last_match_mat[i-1][j-2]

				elif _min == conn2_cum_cost:
					self.cost_mat[i][j] = conn2_cum_cost
					self.op_mat[i][j] = self.CONNECT2
					self.last_match_mat[i][j] = self.last_match_mat[i-2][j-1]

	def init_tables(self):
		'''
		Initializes the cost_matrix and the operations_matrix (back pointers)
		'''
		self.last_match_mat = [ [None] * (len(self.lines2) + 1) for _ in xrange(len(self.lines1) + 1)]
		self.op_mat = [ [self.EMPTY] * (len(self.lines2) + 1) for _ in xrange(len(self.lines1) + 1)]
		self.op_mat[0][0] = self.START
		self.cost_mat = [ [0] * (len(self.lines2) + 1) for _ in xrange(len(self.lines1) + 1)]
		for j in xrange(1, len(self.lines2) + 1):
			self.op_mat[0][j] = self.DEL2
			self.cost_mat[0][j] = self.cost_mat[0][j-1] + self.indel_cost(self.lines2[j-1])

	def get_similarity(self):
		self._table_check()
		total1 = sum(map(lambda line: line.length * line.count, self.lines1))
		total2 = sum(map(lambda line: line.length * line.count, self.lines2))
		_max_dist = total1 + total2  # all indels
		return 1.0 - self.cost_mat[-1][-1] / _max_dist

	def get_matches(self):
		ops = self.get_operations()
		ops = filter(lambda tup: tup[0] == self.PERFECT, ops)
		return map(lambda tup: tup[1:-1], ops)

	def get_operations(self):
		'''
		:return: list( (OPERATION, *lines) )
		'''
		self._table_check()
		ops = list()
		i = len(self.lines1)
		j = len(self.lines2)
		while self.op_mat[i][j] != self.START:
			if self.op_mat[i][j] == self.PERFECT:
				ops.append( (self.PERFECT, self.lines1[i-1], self.lines2[j-1], 0) )
				i -= 1
				j -= 1
			elif self.op_mat[i][j] == self.DEL1:
				ops.append( (self.DEL1, self.lines1[i-1], self.indel_cost(self.lines1[i-1])) )
				i -= 1
			elif self.op_mat[i][j] == self.DEL2:
				ops.append( (self.DEL2, self.lines2[j-1], self.indel_cost(self.lines2[j-1])) )
				j -= 1
			elif self.op_mat[i][j] == self.CONNECT1:
				ops.append( (self.CONNECT1, self.lines1[i-1], self.lines2[j-1], self.lines2[j-2], self.connect_cost(self.lines1[i-1], self.lines2[j-1], self.lines2[j-2])) )
				i -= 1
				j -= 2
			elif self.op_mat[i][j] == self.CONNECT2:
				ops.append( (self.CONNECT2, self.lines2[j-1], self.lines1[i-1], self.lines1[i-2], self.connect_cost(self.lines2[j-1], self.lines1[i-1], self.lines1[i-2])) )
				i -= 2
				j -= 1
			elif self.op_mat[i][j] == self.CONTAINS1:
				# Overlap, contains, etc
				ops.append( (self.CONTAINS1, self.lines1[i-1], self.lines2[j-1], self.contains_cost(self.lines1[i-1], self.lines2[j-1])) )
				i -= 1
				j -= 1
			elif self.op_mat[i][j] == self.CONTAINS2:
				# Overlap, contains, etc
				ops.append( (self.CONTAINS2, self.lines1[i-1], self.lines2[j-1], self.contains_cost(self.lines2[j-1], self.lines1[i-1])) )
				i -= 1
				j -= 1
			elif self.op_mat[i][j] == self.OVERLAP:
				# Overlap, contains, etc
				ops.append( (self.OVERLAP, self.lines1[i-1], self.lines2[j-1], self.overlap_cost(self.lines1[i-1], self.lines2[j-1])) )
				i -= 1
				j -= 1
			elif self.op_mat[i][j] == self.COFRAG:
				# Overlap, contains, etc
				ops.append( (self.COFRAG, self.lines1[i-1], self.lines2[j-1], self.cofrag_cost(self.lines1[i-1], self.lines2[j-1])) )
				i -= 1
				j -= 1
			else:
				assert False  # NO_MATCH?
		ops.reverse()
		return ops

	def combine_perfect(self, op_tup, last_perfect_match):
		line1 = op_tup[1]
		line2 = op_tup[2]
		weights = [line1.count, line2.count]
		if last_perfect_match is None:
			avg_pos = utils.tup_avg([line1.pos, line2.pos], weights)
			avg_len = utils.wavg([line1.length, line2.length], weights)
			avg_thick = utils.wavg([line1.thickness, line2.thickness], weights)
		else:
			# take the average distance from the previous match
			# use last combined line to get absolute pos
			prev1, prev2, prev_comb = last_perfect_match
			pos_diff1 = utils.tup_diff(line1.pos, prev1.pos)
			pos_diff2 = utils.tup_diff(line2.pos, prev2.pos)
			avg_pos_diff = utils.tup_avg([pos_diff1, pos_diff2], weights)

			avg_pos = utils.tup_sum([prev_comb.pos, avg_pos_diff])
			avg_len = utils.wavg([line1.length, line2.length], weights)
			avg_thick = utils.wavg([line1.thickness, line2.thickness], weights)

		combined_line = components.Line(line1.orien, avg_pos, avg_len, avg_thick)
		combined_line.count = line1.count + line2.count
		return combined_line

	def combine_del_helper(self, del_line, prev_line, reference_line):
		pos_diff = utils.tup_diff(del_line.pos, prev_line.pos)
		pos = utils.tup_sum([reference_line.pos, pos_diff])
		line = components.Line(del_line.orien, pos, del_line.length, del_line.thickness)
		line.count = del_line.count
		return line

	def combine_del1(self, op_tup, last_perfect_match):
		line1 = op_tup[1]
		prev1, prev_comb = last_perfect_match[0], last_perfect_match[2]
		return self.combine_del_helper(line1, prev1, prev_comb)

	def combine_del2(self, op_tup, last_perfect_match):
		line2 = op_tup[1]
		prev2, prev_comb = last_perfect_match[1], last_perfect_match[2]
		return self.combine_del_helper(line2, prev2, prev_comb)

	def combine_connect_helper(self, connector, line1, line2, prev_connector, prev_line, prev_combined):
		'''
		:param connector: line - connector line to be merged with line1, line2
		:param line1: line - fragmented line to be merged with connector.  line1.pos is closer to connector than line2.pos is
		:param line2: line - other fragmented line to be merged with connector.
		:param prev_connector: line - previous line in last perfect match.  Comes from same sequence as connector.
		:param prev_line: line - previous line in the last perfect match.  Comes from same sequence as line1/2
		:parm prev_combined: line - prev_connector and prev_line combined.  Gives reference coordinates
		'''
		o = connector.orien
		weights = [connector.count, line1.count, line2.count]

		# dist - weighted average of all three distances to the previous lines
		conn_dist = connector.pos[o] - prev_connector.pos[o]
		line1_dist = line1.pos[o] - prev_line.pos[o]
		line2_dist = line2.pos[o] - prev_line.pos[o]
		avg_dist = utils.wavg([conn_dist, line1_dist, line2_dist], weights)
		o_pos = prev_combined.pos[o] + avg_dist

		# offset - weighted average of two offsets from the previous lines
		conn_offset = connector.pos[1-o] - prev_connector.pos[1-o]
		line1_offset = line1.pos[1-o] - prev_line.pos[1-o]
		avg_offset = utils.wavg([conn_offset, line1_offset], weights[0:2])

		# pos - combine relative dist and offset with the reference combined line
		pos = [0] * 2
		pos[o] = prev_combined.pos[o] + avg_dist
		pos[1-o] = prev_combined.pos[1-o] + avg_offset
		pos = tuple(pos)

		# thickness - weighted average
		avg_thickness = utils.wavg([connector.thickness, line1.thickness, line2.thickness], weights)

		# length - treat the length of the fragments as if they were not fragmented.  Weighted average
		len12 = line2.length + (line2.pos[1-o] - line1.pos[1-o])
		w12 = (line1.count + line2.count) / 2.0
		avg_len = utils.wavg([connector.length, len12], [connector.count, w12])
		
		# count - use pixel mass to determine count
		line = components.Line(o, pos, avg_len, avg_thickness)
		line.count = ((connector.count * connector.length) + (line1.count * line1.length) + (line2.count * line2.length)) / line.length
		return line

	def combine_connect1(self, op_tup, last_perfect_match):
		connector = op_tup[1]
		line1 = op_tup[2]
		line2 = op_tup[3]
		o = line1.orien
		if line1.pos[1-o] > line2.pos[1-o]:
			line1, line2 = line2, line1
		prev_connector, prev_line, prev_combined = last_perfect_match
		return self.combine_connect_helper(connector, line1, line2, prev_connector, prev_line, prev_combined)

	def combine_connect2(self, op_tup, last_perfect_match):
		connector = op_tup[1]
		line1 = op_tup[2]
		line2 = op_tup[3]
		o = line1.orien
		if line1.pos[1-o] > line2.pos[1-o]:
			line1, line2 = line2, line1
		prev_line, prev_connector, prev_combined = last_perfect_match
		return self.combine_connect_helper(connector, line1, line2, prev_connector, prev_line, prev_combined)

	def combine_contains_helper(self, container, containee, prev_container, prev_containee, prev_combined):
		o = container.orien
		weights = [container.count, containee.count]

		# dist
		container_dist = container.pos[o] - prev_container.pos[o]
		containee_dist = containee.pos[o] - prev_containee.pos[o]
		avg_dist = utils.wavg([container_dist, containee_dist], weights)

		# offset
		offset = container.pos[1-o] - prev_container.pos[1-o]

		# pos - combine relative dist and offset with the reference combined line
		pos = [0] * 2
		pos[o] = prev_combined.pos[o] + avg_dist
		pos[1-o] = prev_combined.pos[1-o] + offset
		pos = tuple(pos)

		# thickness
		thickness = utils.wavg([container.thickness, containee.thickness], weights)

		# count - by pixel mass
		line = components.Line(o, pos, container.length, thickness)
		line.count = ((container.count * container.length) + (containee.count * containee.length)) / line.length
		return line

	def combine_contains1(self, op_tup, last_perfect_match):
		container = op_tup[1]
		containee = op_tup[2]
		prev_container, prev_containee, prev_combined = last_perfect_match
		return self.combine_contains_helper(container, containee, prev_container, prev_containee, prev_combined)

	def combine_contains2(self, op_tup, last_perfect_match):
		container = op_tup[1]
		containee = op_tup[2]
		prev_containee, prev_container, prev_combined = last_perfect_match
		return self.combine_contains_helper(container, containee, prev_container, prev_containee, prev_combined)

	def combine_overlap(self, op_tup, last_perfect_match):
		line1 = op_tup[1]
		line2 = op_tup[2]
		prev1 = last_perfect_match[0]
		prev2 = last_perfect_match[1]
		prev_combined = last_perfect_match[2]

		o = line1.orien
		if line1.pos[1-o] > line2.pos[1-o]:
			line1, line2 = line2, line1
			prev1, prev2 = prev2, prev1

		weights = [line1.count, line2.count]

		# dist
		line1_dist = line1.pos[o] - prev1.pos[o]
		line2_dist = line2.pos[o] - prev2.pos[o]
		avg_dist = utils.wavg([line1_dist, line2_dist], weights)

		# offset
		offset = line1.pos[1-o] - prev1.pos[1-o]

		# pos - combine relative dist and offset with the reference combined line
		pos = [0] * 2
		pos[o] = prev_combined.pos[o] + avg_dist
		pos[1-o] = prev_combined.pos[1-o] + offset
		pos = tuple(pos)

		# thickness
		thickness = utils.wavg([line1.thickness, line2.thickness], weights)

		# length -- this has got to be done in a better way (takes count into account)
		length = line2.length + line2.pos[1-o] - line1.pos[1-o]

		# count
		line = components.Line(o, pos, length, thickness)
		line.count = ((line1.count * line1.length) + (line2.count * line2.length)) / line.length
		return line

	def get_merged_lines(self) :
		ops = self.get_operations()
		merged_lines = list()
		last_perfect_match = None

		# get the first perfect match
		for op_tup in ops:
			if op_tup[0] == self.PERFECT:
				combined_line = self.combine_perfect(op_tup, last_perfect_match)
				last_perfect_match = (op_tup[1], op_tup[2], combined_line)
				break

		# error case.  In this case the lines should not be merged at all.
		if last_perfect_match is None:
			#raise Exception("No Perfect Matches in the line sequence")
			return map(lambda line: line.copy(), self.lines1)

		# merge based on each operations tuple
		for x, op_tup in enumerate(ops):
			op = op_tup[0]
			if op == self.PERFECT:
				if op_tup[1] is last_perfect_match[0]:
					# first perfect match already done above.
					line = last_perfect_match[2]
				else:
					line = self.combine_perfect(op_tup, last_perfect_match)
				#last_perfect_match = (op_tup[1], op_tup[2], line)

			elif op == self.DEL1:
				line = self.combine_del1(op_tup, last_perfect_match)

			elif op == self.DEL2:
				line = self.combine_del2(op_tup, last_perfect_match)

			elif op == self.CONNECT1:
				line = self.combine_connect1(op_tup, last_perfect_match)

			elif op == self.CONNECT2:
				line = self.combine_connect2(op_tup, last_perfect_match)

			elif op == self.CONTAINS1:
				line = self.combine_contains1(op_tup, last_perfect_match)

			elif op == self.CONTAINS2:
				line = self.combine_contains2(op_tup, last_perfect_match)

			elif op == self.OVERLAP:
				line = self.combine_overlap(op_tup, last_perfect_match)

			elif op == self.COFRAG:
				line1 = self.combine_del_helper(op_tup[1], last_perfect_match[0], last_perfect_match[2])
				line2 = self.combine_del_helper(op_tup[2], last_perfect_match[1], last_perfect_match[2])
				merged_lines.append(line1)
				line = line2
			else:
				assert False
			if line is None:
				raise Exception("Got back None line:\n" + repr(op_tup) + "\n" + repr(last_perfect_match))

			#print "\t%d: %s %s" % (x, self.op_str(op_tup[0]), op_tup[-1])
			#for l in op_tup[1:-1]:
			#	print "\t\t", l
			#print "\t\tCombined:", line
			#print
			merged_lines.append(line)
		sort_lines(merged_lines)
		return merged_lines

	def print_ops(self, ops):
		'''
		Prints the series of operations nicely
		'''
		print "Sequence of operations:"
		for x, op in enumerate(ops):
			print "\t%d: %s %s" % (x, self.op_str(op[0]), op[-1])
			for line in op[1:-1]:
				print "\t\t", line
			print
		print

	def are_colinear(self, line1, line2):
		o = line1.orien
		if abs(line1.pos[o] - line2.pos[o]) > self.dist_thresh:
			return False
		r1 = line1.length_range()
		r2 = line2.length_range()
		return not utils.are_overlapping([r1, r2])

	def display(self):
		'''
		prints the cost_mat and the op_mat
		'''
		print "Cost Matrix"
		utils.print_mat(self.cost_mat)
		print
		print "Operations Matrix"
		utils.print_mat(self.op_mat)
		print

