
import xml.etree.ElementTree as ET
import ImageDraw
import Image
import math

import components
import utils
#from profilehooks import profile

def create_pos(dist, offset, o):
	pos = [0] * 2
	pos[o] = dist
	pos[1-o] = offset
	pos = tuple(pos)
	return pos

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


class LineMatcher():
	
	def __init__(self, lines1, lines2):
		'''
		:param lines1: list of Lines
		:param lines2: list of Lines
		'''
		# makes sure that we have at most one orientation in both lists
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
		'''
		sort_lines(self.lines1)
		sort_lines(self.lines2)


class LMatcher(LineMatcher):
	'''
	Uses dyanmic programming like edit distance with operations specific to lines.
	After the edit distance is computed, the algorithm looks for transposes.
	The main idea is to slightly greedily detect the perfect matches, which serve
	as reference points for dealing with the coordinate offset problem.  Then we
	fine tune and match as many things as possible.  Also this class allows for
	line sequence merging based on partial matches.
	Merging is asymmetrical; lines1 is perturbed by lines2
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
	TRANSPOSE = 9            # a pair of transposed PERFECT matchings
	NO_MATCH = 10            # indicates that line1 and line2 do not match any of the above cases
	NO_MATCH_COST = 1e10     # Arbitrary high cost to cause certain operations to be never chosen as optimal

	OP_STR = {EMPTY: "Empty", START: "Start", PERFECT: "Perfect", CONTAINS1: "Contains1",
				 CONTAINS2: "Contains2", OVERLAP: "Overlap", CONNECT1: "Connect1", CONNECT2: "Connect2",
				 COFRAG: "Cofrag", TRANSPOSE: "Transpose", DEL1: "Del1", DEL2: "Del2", NO_MATCH: "No_Match"}

	def __init__(self, lines1, lines2, dist_thresh, size):
		LineMatcher.__init__(self, lines1, lines2)
		self.size = size
		self.dist_thresh = dist_thresh
		self.offset_thresh = dist_thresh / 2
		self.colinear_thresh = dist_thresh / 5.0
		self.tables_built = False
		self.match_cost_table = [[None] * (len(self.lines2)+1) for i in range(len(self.lines1)+1)]
		self.sort()

	def _table_check(self):
		'''
		For lazy operations
		'''
		if not self.tables_built:
			self.build_tables()
		self.tables_built = True

	def op_str(self, op):
		return self.OP_STR.get(op)

	def indel_cost(self, line):
		'''
		Shift invariant
		Cost of deleting/inserting a line.  Linear to the number of times a line occurs.
		Biases algorithm to match lines with a large count.
		'''
		return line.length * line.count

	def contains_cost(self, container, containee):
		'''
		Cost of line1 containing line2
		Biases algorithm to match lines with a large count.
		Shift invariant
		:param container: containing line
		:param containee: contained line
		'''
		return (container.length - containee.length) * (container.count + containee.count) / 2

	def connect_cost(self, connector, line1, line2, pos_offset=None):
		'''
		Cost of connector connecting line1 and line2
		Assumes the lines are approx colinear
		:param connector: line connecting line1/2
		:param line1/2: line
		:param offset: pos offset: connector.pos coords = line1/2.pos coords + offset
		'''
		if pos_offset is None:
			pos_offset = (0, 0)
		o = connector.orien
		if (line1.pos[1-o] > line2.pos[1-o]):
			# make sure the line1 has the smaller position
			line1, line2 = line2, line1

		connector_range = connector.length_range()
		line1_range = line1.length_range(offset=pos_offset[1-o])
		line2_range = line2.length_range(offset=pos_offset[1-o])

		gap_len = utils.gap_len(line1_range, line2_range)
		cost = gap_len * (connector.count + ( (line1.count + line2.count) / 2)) # double count the gap

		# count the cost if line1 and line2 don't align perfectly with the ends of connector
		start_diff = abs(connector_range[0] - line1_range[0])
		cost += start_diff * ( (connector.count + line1.count) / 2.0)
		end_diff = abs(connector_range[1] - line2_range[1])
		cost += end_diff * ( (connector.count + line2.count) / 2.0)
		return cost

	def cofrag_cost(self, line1, line2):
		'''
		Cost of saying line1 and line2 are cofragments
		Bias is toward this operation over two successive indels
		Shift invarient
		'''
		return (self.indel_cost(line1) + self.indel_cost(line2) - 1)

	def overlap_cost(self, line1, line2, pos_offset=None):
		'''
		Cost of saying line1 and line2 overlap
		:param offset: pos offset: line1.pos coords = line2.pos coords + offset
		'''
		if pos_offset is None:
			pos_offset = (0, 0)
		avg_count = (line1.count + line2.count) / 2.0 
		o = line1.orien
		line1_range = line1.length_range()
		line2_range = line2.length_range(offset=pos_offset[1-o])
		o_len = utils.overlap_len(line1_range, line2_range)
		return avg_count * (line1.length + line2.length - 2 * o_len)


	def cached_match_cost_type(self,i,j):
		if (self.match_cost_table[i][j] == None):
			tmp = self.match_cost_type(i,j)
			self.match_cost_table[i][j] = tmp
			return tmp	
		
		return self.match_cost_table[i][j]
			

	LEN_RATIO_THRESH = 1.20
	def match_cost_type(self, i, j):
			
		line1 = self.lines1[i-1]
		line2 = self.lines2[j-1]
		o = line1.orien
		pos_offset = self.global_offsets[i-1][j-1]
		len_ratio = max(line1.length / float(line2.length), line2.length / float(line1.length))

		# calculate distance and offset
		if pos_offset:
			dist = abs(line1.pos[o] - line2.pos[o] - pos_offset[o])
			offset = abs(line1.pos[1-o] - line2.pos[1-o] - pos_offset[1-o])
		else:
			dist = abs(line1.pos[o] - line2.pos[o])
			offset = abs(line1.pos[1-o] - line2.pos[1-o])
			if len_ratio < self.LEN_RATIO_THRESH:
				# small penaly for large divergence
				return (dist + offset) * line1.count, self.PERFECT
			else:
				return self.NO_MATCH_COST, self.NO_MATCH

		# anything not the proper distance away is not considered
		if dist > self.dist_thresh:
			return self.NO_MATCH_COST, self.NO_MATCH

		if offset < self.offset_thresh and len_ratio < self.LEN_RATIO_THRESH:
			return 0, self.PERFECT


		line1_range = line1.length_range()
		line2_range = line2.length_range(offset=pos_offset[1-o])
		if utils.range_contains(line1_range, line2_range):
			return self.contains_cost(line2, line1), self.CONTAINS2

		if utils.range_contains(line2_range, line1_range):
			return self.contains_cost(line1, line2), self.CONTAINS1

		if utils.are_overlapping([line1_range, line2_range]):
			return self.overlap_cost(line1, line2, pos_offset), self.OVERLAP

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

		offset = self.global_offsets[i-1][j-2]
		if offset is None:
			offset = (0, 0)
		o = line1.orien
		r1 = line1.length_range()
		r21 = line21.length_range(offset=offset[1-o])
		r22 = line22.length_range(offset=offset[1-o])
		if self.are_colinear(line21, line22) and utils.are_overlapping([r1, r21, r22]):
			return self.connect_cost(line1, line21, line22, offset), self.CONNECT1
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

		offset = self.global_offsets[i-2][j-1]
		if offset is None:
			offset = (0, 0)
		o = line2.orien

		r2 = line2.length_range(offset=offset[1-o])
		r11 = line11.length_range()
		r12 = line12.length_range()

		if self.are_colinear(line11, line12) and utils.are_overlapping([r2, r11, r12]):
			return self.connect_cost(line2, line11, line12, utils.tup_scale(offset, -1)), self.CONNECT2
		else:
			return self.NO_MATCH_COST, self.NO_MATCH

	def transpose_cost(self, i, j):
		# make sure that we don't make inconsistent perfect matches
		if i < 2 or j < 2 or self.global_offsets[i-2][j-1] is None or self.global_offsets[i-1][j-2] is None:
			return self.NO_MATCH_COST, self.NO_MATCH
		cost1, type1 = self.cached_match_cost_type(i, j-1)
		cost2, type2 = self.cached_match_cost_type(i-1, j)
		if type1 == self.PERFECT and type2 == self.PERFECT:
			return 0, self.TRANSPOSE
		else:
			return self.NO_MATCH_COST, self.NO_MATCH

	#@profile(sort='tottime')
	def build_tables(self):
		self.init_tables()
		for i in xrange(1, len(self.lines1) + 1):
			self.op_mat[i][0] = self.DEL1
			self.cost_mat[i][0] = self.cost_mat[i-1][0] + self.indel_cost(self.lines1[i-1])
			for j in xrange(1, len(self.lines2) + 1):
				# marginal costs
				del1_marg_cost = self.indel_cost(self.lines1[i-1])
				del2_marg_cost = self.indel_cost(self.lines2[j-1])
				match_marg_cost, match_type = self.cached_match_cost_type(i, j)
				conn1_marg_cost, conn1_type = self.connect1_cost(i, j) if match_type in [self.OVERLAP, self.CONTAINS1] else (self.NO_MATCH_COST, self.NO_MATCH)
				conn2_marg_cost, conn2_type = self.connect2_cost(i, j) if match_type in [self.OVERLAP, self.CONTAINS2] else (self.NO_MATCH_COST, self.NO_MATCH)
				transpose_marg_cost, transpose_type = self.transpose_cost(i, j)

				# cumulative costs
				del1_cum_cost = del1_marg_cost + self.cost_mat[i-1][j]
				del2_cum_cost = del2_marg_cost + self.cost_mat[i][j-1]
				match_cum_cost = match_marg_cost + self.cost_mat[i-1][j-1]
				conn1_cum_cost = conn1_marg_cost + (self.cost_mat[i-1][j-2] if j > 1 else self.NO_MATCH_COST)
				conn2_cum_cost = conn2_marg_cost + (self.cost_mat[i-2][j-1] if i > 1 else self.NO_MATCH_COST)
				transpose_cum_cost = transpose_marg_cost + (self.cost_mat[i-2][j-2] if (j > 1 and i > 1) else self.NO_MATCH_COST)

				# update tables
				_min = min(del1_cum_cost, del2_cum_cost, match_cum_cost, conn1_cum_cost, conn2_cum_cost, transpose_cum_cost)
				if _min == del1_cum_cost:
					self.cost_mat[i][j] = del1_cum_cost
					self.op_mat[i][j] = self.DEL1
					self.global_offsets[i][j] = self.global_offsets[i-1][j]

				elif _min == del2_cum_cost:
					self.cost_mat[i][j] = del2_cum_cost
					self.op_mat[i][j] = self.DEL2
					self.global_offsets[i][j] = self.global_offsets[i][j-1]

				elif _min == match_cum_cost:
					self.cost_mat[i][j] = match_cum_cost
					self.op_mat[i][j] = match_type
					if match_type == self.PERFECT and self.global_offsets[i-1][j-1] is None:
						self.global_offsets[i][j] = utils.tup_diff(self.lines1[i-1].pos, self.lines2[j-1].pos)
					else:
						self.global_offsets[i][j] = self.global_offsets[i-1][j-1]

				elif _min == conn1_cum_cost:
					self.cost_mat[i][j] = conn1_cum_cost
					self.op_mat[i][j] = self.CONNECT1
					self.global_offsets[i][j] = self.global_offsets[i-1][j-2]

				elif _min == conn2_cum_cost:
					self.cost_mat[i][j] = conn2_cum_cost
					self.op_mat[i][j] = self.CONNECT2
					self.global_offsets[i][j] = self.global_offsets[i-2][j-1]

				elif _min == transpose_cum_cost:
					self.cost_mat[i][j] = transpose_cum_cost
					self.op_mat[i][j] = self.TRANSPOSE
					self.global_offsets[i][j] = self.global_offsets[i-2][j-2]

	def init_tables(self):
		'''
		Initializes the cost_matrix and the operations_matrix (back pointers)
		'''
		# line1 coords = line2 coords + global offset
		self.global_offsets = [ [None] * (len(self.lines2) + 1) for _ in xrange(len(self.lines1) + 1)]
		self.op_mat = [ [self.EMPTY] * (len(self.lines2) + 1) for _ in xrange(len(self.lines1) + 1)]
		self.op_mat[0][0] = self.START
		self.cost_mat = [ [0] * (len(self.lines2) + 1) for _ in xrange(len(self.lines1) + 1)]
		for j in xrange(1, len(self.lines2) + 1):
			self.op_mat[0][j] = self.DEL2
			self.cost_mat[0][j] = self.cost_mat[0][j-1] + self.indel_cost(self.lines2[j-1])

	def _get_region(self, pos, width, height):
		row = int(pos[1]) / height
		col = int(pos[0]) / width
		return (row, col)

	def _get_regions(self, line, width, height):
		r, c = self._get_region(line.pos, width, height)
		regions = list()
		if line.is_horizontal():
			
			# first region is partial
			first_len = width - (line.pos[0] % width)
			regions.append( (r, c, min(first_len, line.length) / float(line.length)) )
			c += 1
			remaining = line.length - first_len

			# middle regions are complete
			while remaining >= width:
				regions.append( (r, c, width / float(line.length)) )
				remaining -= width
				c += 1
			# last region may be partial or even empty
			if remaining > 0:
				regions.append( (r, c, remaining / float(line.length)) )
		else:
			first_len = height - (line.pos[1] % height)
			regions.append( (r, c, min(first_len, line.length) / float(line.length)) )
			r += 1
			remaining = line.length - first_len

			# middle regions are complete
			while remaining >= height:
				regions.append( (r, c, height / float(line.length)) )
				remaining -= height
				r += 1
			# last region may be partial or even empty
			if remaining > 0:
				regions.append( (r, c, remaining / float(line.length)) )

		return regions

	def _update_region_mats(self, line, actual_cost, total_mat, actual_mat, width, height, rows, cols):
		# note that the matching cost can exceed the indel cost of the one line
		# I don't expect that to happen often because the prototype lines have high counts
		#print line
		#print "\t", width, height
		#print "\t", actual_cost 
		del_cost = max(actual_cost, self.indel_cost(line))
		regions = self._get_regions(line, width, height)
		#for region in regions:
			#print "\t", region
		for r, c, p in regions:
			if r >= rows or c >= cols or r < 0 or c < 0:
				continue
			total_mat[r][c] += p * del_cost
			actual_mat[r][c] += p * actual_cost

	def similarity_by_region(self, rows, cols, size):
		'''
		:param rows: int number of rows
		:param cols: int number of cols
		:param size: (int, int) size of image1
		:return: list(list(float(0-1))) matrix of regional percentage matches
		'''
		#print size
		#print rows, cols
		ops = self.get_operations()
		width = (size[0] / cols) + 1
		height = (size[1] / rows) + 1
		total_cost_mat = [([0] * cols) for r in xrange(rows)]
		actual_cost_mat = [([0] * cols) for r in xrange(rows)]
		for op_tup in ops:
			op = op_tup[0]
			if op in  [self.PERFECT, self.DEL1, self.CONTAINS1, self.CONTAINS2, 
					   self.CONNECT1, self.OVERLAP, self.COFRAG]:
				line1 = op_tup[1]
				actual_cost = op_tup[-1]
				self._update_region_mats(line1, actual_cost, total_cost_mat, actual_cost_mat, width, height, rows, cols)

			elif op == self.DEL2:
				pass

			elif op == self.CONNECT2:
				line11 = op_tup[2]
				line12 = op_tup[3]
				actual_cost = op_tup[-1]
				# split responsibility down the middle
				self._update_region_mats(line11, actual_cost / 2, total_cost_mat, actual_cost_mat, width, height, rows, cols)
				self._update_region_mats(line12, actual_cost / 2, total_cost_mat, actual_cost_mat, width, height, rows, cols)

			elif op == self.TRANSPOSE:
				line11 = op_tup[1]
				line12 = op_tup[2]
				actual_cost = op_tup[-1]
				# split responsibility down the middle
				self._update_region_mats(line11, actual_cost / 2, total_cost_mat, actual_cost_mat, width, height, rows, cols)
				self._update_region_mats(line12, actual_cost / 2, total_cost_mat, actual_cost_mat, width, height, rows, cols)
			else:
				assert False

		perc_mat = [([0] * cols) for r in xrange(rows)]
		total = 0
		for r in xrange(rows):
			for c in xrange(cols):
				perc_mat[r][c] = 1 - actual_cost_mat[r][c] / total_cost_mat[r][c] if total_cost_mat[r][c] else 0 #float('NaN')
				total += total_cost_mat[r][c]
		weight_mat = [([0] * cols) for r in xrange(rows)]
		for r in xrange(rows):
			for c in xrange(cols):
				weight_mat[r][c] = total_cost_mat[r][c] / total if total else 0
		return perc_mat, weight_mat

	def similarity(self):
		self._table_check()
		total1 = sum(map(lambda line: line.length * line.count, self.lines1))
		total2 = sum(map(lambda line: line.length * line.count, self.lines2))
		_max_dist = float(total1 + total2)  # all indels
		#print self.cost_mat[-1][-1]
		#print _max_dist
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
			op = self.op_mat[i][j]  
			if op == self.PERFECT:
				ops.append( (self.PERFECT, self.lines1[i-1], self.lines2[j-1], self.cost_mat[i][j] - self.cost_mat[i-1][j-1]) )
				i -= 1
				j -= 1
			elif op == self.DEL1:
				ops.append( (self.DEL1, self.lines1[i-1], self.cost_mat[i][j] - self.cost_mat[i-1][j]) )
				i -= 1
			elif op == self.DEL2:
				ops.append( (self.DEL2, self.lines2[j-1], self.cost_mat[i][j] - self.cost_mat[i][j-1]) )
				j -= 1
			elif op == self.CONNECT1:
				ops.append( (self.CONNECT1, self.lines1[i-1], self.lines2[j-1], self.lines2[j-2], self.cost_mat[i][j] - self.cost_mat[i-1][j-2]) )
				i -= 1
				j -= 2
			elif op == self.CONNECT2:
				ops.append( (self.CONNECT2, self.lines2[j-1], self.lines1[i-1], self.lines1[i-2], self.cost_mat[i][j] - self.cost_mat[i-2][j-1]) )
				i -= 2
				j -= 1
			elif op == self.CONTAINS1:
				ops.append( (self.CONTAINS1, self.lines1[i-1], self.lines2[j-1], self.cost_mat[i][j] - self.cost_mat[i-1][j-1]) )
				i -= 1
				j -= 1
			elif op == self.CONTAINS2:
				ops.append( (self.CONTAINS2, self.lines1[i-1], self.lines2[j-1], self.cost_mat[i][j] - self.cost_mat[i-1][j-1]) )
				i -= 1
				j -= 1
			elif op == self.OVERLAP:
				ops.append( (self.OVERLAP, self.lines1[i-1], self.lines2[j-1], self.cost_mat[i][j] - self.cost_mat[i-1][j-1]) )
				i -= 1
				j -= 1
			elif op == self.COFRAG:
				ops.append( (self.COFRAG, self.lines1[i-1], self.lines2[j-1], self.cost_mat[i][j] - self.cost_mat[i-1][j-1]) )
				i -= 1
				j -= 1
			elif op == self.TRANSPOSE:
				ops.append( (self.TRANSPOSE, self.lines1[i-2], self.lines2[j-1], (self.cost_mat[i][j] - self.cost_mat[i-2][j-2]) / 2.0) )
				ops.append( (self.TRANSPOSE, self.lines1[i-1], self.lines2[j-2], (self.cost_mat[i][j] - self.cost_mat[i-2][j-2]) / 2.0) )
				i -= 2
				j -= 2
			else:
				print "ERROR"
				self.print_ops(ops)
				print op

				print "First sequence"
				for line1 in self.lines1:
					print line1

				print "Second sequence"
				for line2 in self.lines2:
					print line2

				assert False  # NO_MATCH?
		ops.reverse()
		return ops

	def combine_perfect(self, line1, line2, pos_offset):

		weights = [line1.count, line2.count]
		pos2_translated = utils.tup_sum([line2.pos, pos_offset]) 
		avg_pos = utils.tup_avg([line1.pos, pos2_translated], weights)
		avg_len = max(utils.wavg([line1.length, line2.length], weights), 0.1)
		avg_thick = utils.wavg([line1.thickness, line2.thickness], weights)

		combined_line = components.Line(line1.orien, avg_pos, avg_len, avg_thick)
		combined_line.count = line1.count + line2.count
		return combined_line

	def combine_del(self, del_line, pos_offset):
		pos = utils.tup_sum([del_line.pos, pos_offset])
		line = components.Line(del_line.orien, pos, del_line.length, del_line.thickness)
		line.count = del_line.count
		return line

	def interpolate_lines(self, line1, line2):
		'''
		Connects two lines.  Takes their average distance, min of the offset, and fills the gap for length
		'''
		o = line1.orien
		if line1.pos[1-o] > line2.pos[1-o]:
			line1, line2 = line2, line1

		weights = [line1.count, line2.count]
		avg_dist = utils.wavg([line1.pos[o], line2.pos[o]], weights)
		offset = line1.pos[1-o]
		pos = create_pos(avg_dist, offset, o)
		length = max(line2.length + (line2.pos[1-o] - line1.pos[1-o]), 0.1)
		avg_thickness = utils.wavg([line1.thickness, line2.thickness], weights)

		line = components.Line(o, pos, length, avg_thickness)
		line.count = (line1.count * line1.length + line2.count * line2.length) / float(line.length)
		return line

	def combine_connect1(self, op_tup, pos_offset):
		'''
		Combines the fragmented lines into one, then treats the operation as a CONTAINS or OVERLAP
		'''
		connector = op_tup[1]  # from sequence 1
		combined_frags = self.interpolate_lines(op_tup[2], op_tup[3])
		o = connector.orien

		connector_range = connector.length_range()
		frags_range = combined_frags.length_range(offset=pos_offset[1-o])

		if utils.range_contains(frags_range, connector_range):
			return self.combine_contains1( (None, connector, combined_frags) , pos_offset)
		elif utils.range_contains(connector_range, frags_range):
			return self.combine_contains2( (None, connector, combined_frags) , pos_offset)
		else:
			return self.combine_overlap( (None, connector, combined_frags) , pos_offset)

	def combine_connect2(self, op_tup, pos_offset):
		'''
		Treats the so called fragmented lines separately and does contains/overlap operations
		'''
		connector = op_tup[1]  # from sequence 2
		line1 = op_tup[2]
		line2 = op_tup[3]
		o = connector.orien

		connector_range = connector.length_range(offset=pos_offset[1-o])
		line1_range = line1.length_range()

		if utils.range_contains(line1_range, connector_range):
			combined1 = self.combine_contains2( (None, line1, connector) , pos_offset)
		else:
			combined1 = self.combine_overlap( (None, line1, connector) , pos_offset)

		line2_range = line2.length_range()

		if utils.range_contains(line2_range, connector_range):
			combined2 = self.combine_contains2( (None, line2, connector) , pos_offset)
		else:
			combined2 = self.combine_overlap( (None, line2, connector) , pos_offset)

		return combined1, combined2

	# we need to shrink line1 a bit
	def combine_contains1(self, op_tup, pos_offset):
		line1 = op_tup[1]  # container
		line2 = op_tup[2]  # containee

		o = line1.orien
		weights = [line1.count, line2.count]

		# dist
		line1_dist = line1.pos[o]
		line2_dist = line2.pos[o] + pos_offset[o]
		avg_dist = utils.wavg([line1_dist, line2_dist], weights)

		# length 
		len_diff = line1.length - line2.length
		overlap_perc = line2.length / line1.length
		length_delta = len_diff * overlap_perc / max(line1.count, 1)
		length = max(line1.length - length_delta, 0.1)

		# offset
		offset = line1.pos[1-o] + (length_delta / 2)

		# pos
		pos = create_pos(avg_dist, offset, o)

		# thickness
		thickness = utils.wavg([line1.thickness, line2.thickness], weights)

		# count - by pixel mass
		line = components.Line(o, pos, length, thickness)
		line.count = ((line1.count * line1.length) + (line2.count * line2.length)) / line.length
		return line

	# we need to grow line1 a bit
	def combine_contains2(self, op_tup, pos_offset):
		line1 = op_tup[1]  # containee
		line2 = op_tup[2]  # container

		o = line1.orien
		weights = [line1.count, line2.count]

		# dist
		line1_dist = line1.pos[o]
		line2_dist = line2.pos[o] + pos_offset[o]
		avg_dist = utils.wavg([line1_dist, line2_dist], weights)

		# length 
		len_diff = line2.length - line1.length
		overlap_perc = line1.length / line2.length
		length_delta = len_diff * overlap_perc / max(line1.count, 1)
		length = max(line1.length + length_delta, 0.1)

		# offset
		offset = line1.pos[1-o] - (length_delta / 2)

		# pos
		pos = create_pos(avg_dist, offset, o)

		# thickness
		thickness = utils.wavg([line1.thickness, line2.thickness], weights)

		# count - by pixel mass
		line = components.Line(o, pos, length, thickness)
		line.count = ((line1.count * line1.length) + (line2.count * line2.length)) / line.length
		return line

	def combine_overlap(self, op_tup, pos_offset):
		line1 = op_tup[1]
		line2 = op_tup[2]

		o = line1.orien
		overlap_len = utils.overlap_len(line1.length_range(), line2.length_range(offset=pos_offset[1-o]))
		overlap_perc1 = overlap_len / float(line1.length)
		weights = [line1.count, line2.count]

		# dist
		line1_dist = line1.pos[o]
		line2_dist = line2.pos[o] + pos_offset[o]
		avg_dist = utils.wavg([line1_dist, line2_dist], weights)

		# length
		length_delta = overlap_perc1 * (line2.length - overlap_len) / max(line1.count, 1)
		length = max(line1.length + length_delta, 0.1)

		# decide which end to grow
		if line1.pos[1-o] < (line2.pos[1-o] + pos_offset[1-o]):
			# line1 preceeds line2
			offset = line1.pos[1-o] 
		else:
			# moves line1 toward line2
			offset = line1.pos[1-o] - length_delta

		# pos - combine relative dist and offset with the reference combined line
		pos = create_pos(avg_dist, offset, o)

		# thickness
		thickness = utils.wavg([line1.thickness, line2.thickness], weights)

		# count
		line = components.Line(o, pos, length, thickness)
		line.count = ((line1.count * line1.length) + (line2.count * line2.length)) / line.length
		return line

	def merge(self):
		ops = self.get_operations()

		# compute the average pos offset of the perfect matching pairs
		pos_offsets = list()
		for op_tup in ops:
			if op_tup[0] == self.PERFECT:
				pos_offsets.append(utils.tup_diff(op_tup[1].pos, op_tup[2].pos))
		pos_offset = utils.tup_avg(pos_offsets) if pos_offsets else (0, 0)

		merged_lines = list()
		# merge based on each operations tuple
		for x, op_tup in enumerate(ops):
			op = op_tup[0]
			if op == self.PERFECT or op == self.TRANSPOSE:
				line = self.combine_perfect(op_tup[1], op_tup[2], pos_offset)

			elif op == self.DEL1:
				line = self.combine_del(op_tup[1], (0, 0))

			elif op == self.DEL2:
				line = self.combine_del(op_tup[1], pos_offset)

			elif op == self.CONNECT1:
				line = self.combine_connect1(op_tup, pos_offset)

			elif op == self.CONNECT2:
				line1, line2 = self.combine_connect2(op_tup, pos_offset)
				line1.truncate(self.size)
				merged_lines.append(line1)
				line = line2

			elif op == self.CONTAINS1:
				line = self.combine_contains1(op_tup, pos_offset)

			elif op == self.CONTAINS2:
				line = self.combine_contains2(op_tup, pos_offset)

			elif op == self.OVERLAP:
				line = self.combine_overlap(op_tup, pos_offset)

			elif op == self.COFRAG:
				line1 = self.combine_del(op_tup[1], (0, 0))
				line2 = self.combine_del(op_tup[2], pos_offset)
				line1.truncate(self.size)
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
			line.truncate(self.size)
			merged_lines.append(line)
		sort_lines(merged_lines)
		merged_lines = filter(lambda line: line.length > 1, merged_lines)
		return merged_lines

	def get_match_vector(self):
		ops = self.get_operations()
		vector = list()
		for op_tup in ops:
			op = op_tup[0]
			if op == self.PERFECT or op == self.TRANSPOSE:
				vector.append(1)

			elif op in [self.DEL1, self.COFRAG]:
				vector.append(0)

			elif op == self.DEL2:
				pass

			elif op == self.CONNECT2:
				vector.append( 1 -  (op_tup[4] / 2) / float(self.indel_cost(op_tup[2])))
				vector.append( 1 -  (op_tup[4] / 2) / float(self.indel_cost(op_tup[3])))

			elif op in [self.CONNECT1, self.CONTAINS1, self.CONTAINS2, self.OVERLAP]:
				vector.append(1 - op_tup[-1] / float(self.indel_cost(op_tup[1])))

			else:
				assert False

		assert len(vector) == len(self.lines1)
		return vector

	def push_away(self, perc):
		'''
		Modifies both sequences.  
		If everything matches, they can't get pushed apart
		'''
		ops = self.get_operations()
		matched_weight1 = 0
		unmatched_weight1 = 0
		matched_weight2 = 0
		unmatched_weight2 = 0
		for op_tup in ops:
			op = op_tup[0]
			op_cost = op_tup[-1]
			del_cost = float(sum(map(lambda line: self.indel_cost(line), op_tup[1:-1])))
			match_perc = 1 - (op_cost / del_cost)

			seq1_lines = list()
			seq2_lines = list()
			if op in [self.PERFECT, self.TRANSPOSE, self.CONTAINS1, self.CONTAINS2, self.OVERLAP, self.COFRAG]:
				seq1_lines.append(op_tup[1])
				seq2_lines.append(op_tup[2])
			elif op == self.DEL1:
				seq1_lines.append(op_tup[1])
			elif op == self.DEL2:
				seq2_lines.append(op_tup[1])
			elif op == self.CONNECT1:
				seq1_lines.append(op_tup[1])
				seq2_lines.append(op_tup[2])
				seq2_lines.append(op_tup[3])
			elif op == self.CONNECT2:
				seq2_lines.append(op_tup[1])
				seq1_lines.append(op_tup[2])
				seq1_lines.append(op_tup[3])

			for line1 in seq1_lines:
				line1_weight = self.indel_cost(line1)
				matched_weight1 += line1_weight * match_perc
				unmatched_weight1 += line1_weight * (1 - match_perc)
			for line2 in seq2_lines:
				line2_weight = self.indel_cost(line2)
				matched_weight2 += line2_weight * match_perc
				unmatched_weight2 += line2_weight * (1 - match_perc)

		total_weight1 = matched_weight1 + unmatched_weight1
		total_weight2 = matched_weight2 + unmatched_weight2
		redistribute_weight1 = matched_weight1 * perc
		redistribute_weight2 = matched_weight2 * perc

		#print "\nTotal Weight1: %.2f" % total_weight1
		#print "\tMatched Weight1: %.2f" % matched_weight1
		#print "\tUnMatched Weight1: %.2f" % unmatched_weight1
		#print "\tRedistributed Weight1: %.2f" % redistribute_weight1

		#print "\nTotal Weight2: %.2f" % total_weight2
		#print "\tMatched Weight2: %.2f" % matched_weight2
		#print "\tUnMatched Weight2: %.2f" % unmatched_weight2
		#print "\tRedistributed Weight2: %.2f\n" % redistribute_weight2

		if redistribute_weight1 == 0 or redistribute_weight2 == 0:
			return

		for op_tup in ops:
			op = op_tup[0]
			op_cost = op_tup[-1]
			del_cost = float(sum(map(lambda line: self.indel_cost(line), op_tup[1:-1])))
			match_perc = 1 - (op_cost / del_cost)

			seq1_lines = list()
			seq2_lines = list()
			if op in [self.PERFECT, self.TRANSPOSE, self.CONTAINS1, self.CONTAINS2, self.OVERLAP, self.COFRAG]:
				seq1_lines.append(op_tup[1])
				seq2_lines.append(op_tup[2])
			elif op == self.DEL1:
				seq1_lines.append(op_tup[1])
			elif op == self.DEL2:
				seq2_lines.append(op_tup[1])
			elif op == self.CONNECT1:
				seq1_lines.append(op_tup[1])
				seq2_lines.append(op_tup[2])
				seq2_lines.append(op_tup[3])
			elif op == self.CONNECT2:
				seq2_lines.append(op_tup[1])
				seq1_lines.append(op_tup[2])
				seq1_lines.append(op_tup[3])

			for line1 in seq1_lines:
				match_diff = line1.count * match_perc * perc
				unmatch_diff = (1 - match_perc) * redistribute_weight1 / unmatched_weight1
				line1.count -= match_diff
				line1.count += unmatch_diff
			for line2 in seq2_lines:
				match_diff = line2.count * match_perc * perc
				unmatch_diff = (1 - match_perc) * redistribute_weight2 / unmatched_weight2
				line2.count -= match_diff
				line2.count += unmatch_diff


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

	#TODO: this is pretty hacky and hard to justify
	def are_colinear(self, line1, line2):
		o = line1.orien
		r1 = line1.length_range()
		r2 = line2.length_range()
		if utils.range_contains(r1, r2) or utils.range_contains(r2, r1):
			return False
		overlap_len = float(utils.overlap_len(r1, r2))
		perc = max(overlap_len / line1.length, overlap_len / line2.length)
		if perc > 0.75:
			return False
		return abs(line1.pos[o] - line2.pos[o]) <= self.colinear_thresh

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

