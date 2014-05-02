
import utils
import itertools
import collections


class TextLineMatcher:

	PERFECT = 0
	PARTIAL1 = 1  # line1 matches two lines in lines2
	PARTIAL2 = 2  # line2 matches two lines in lines1
	SUFFIX1 = 3   # line1 is a suffix of line2
	SUFFIX2 = 4   # line2 is a suffix of line1
	PREFIX1 = 5   # line1 is a prefix of line2
	PREFIX2 = 6   # line2 is a prefix of line1
	# prefixes and suffixes are condensed into Partials if they can

	OP_STR = {PERFECT: "Perfect", PARTIAL1: "Partial1", PARTIAL2: "Partial2", SUFFIX1: "Suffix1",
				SUFFIX2: "Suffix2", PREFIX1: "Prefix1", PREFIX2: "Prefix2"}
	
	def __init__(self, lines1, lines2, dist_thresh, partials=False):
		'''
		:param lines1: lines of TextLines
		:param lines2: lines of TextLines
		:param dist_thresh: num distance threshold
		:param partials: bool whether to use partial matches or not
		'''
		self.lines1 = lines1
		self.lines2 = lines2
		self.partial_matches = partials

	def op_str(self, op):
		return self.OP_STR.get(op)
	
	def similarity(self):
		'''
		:return: 0-1 similarity score between the two text lines
		'''
		self.get_matches()  # make sure that lines get matched
		total_val = 0.0
		matched_val = 0.0
		for line in itertools.chain(self.lines1, self.lines2):
			total_val += line.matched_value()
			if line.matched:
				matched_val += line.matched_value()
		return matched_val / total_val

	def _clear_matches(self):
		'''
		Marks all lines as not matched
		'''
		for line in itertools.chain(self.lines1, self.lines2):
			line.matched = False

	def _find_perfect_matches(self):
		self._clear_matches()
		perfect_matches = list()
		for line1 in self.lines1:
			for line2 in self.lines2:
				if line2.matched:
					continue
				if perfect_match(line1, line2):
					perfect_matches.append( (self.PERFECT, line1, line2) )
					line1.matched = True
					line2.matched = True
		return prefect_matches

	SIZE_RATIO = 0.8
	EDIT_DIST_THRESH = 0.2
	def perfect_match(self, line1, line2):
		'''
		A Perfect match is defined to be two lines whose start positions are within
			the distance threshold, whose sizes are similar, and whose normalized
			edit distance is below some threshold
		:param line1: TextLine
		:param line2: TextLine
		:return: bool
		'''
		if utils.e_dist(line1.pos, line2.pos) > self.dist_thresh:
			return False
			
		if not (utils.ratio(line1.size[0], line2.size[0]) > self.SIZE_RATIO and
			utils.ratio(line1.size[1], line2.size[1]) > self.SIZE_RATIO):
			return False

		# optimization using diff in length as a lower bound on edit distance
		if utils.ratio(line1.N, line2.N) < (1 - self.EDIT_DIST_THRESH):
			return False

		# check equality before heavy calculation
		edit_dist = Levenshtein.dist(line1.text, line2.text) if line1.text != line2.text else 0
		norm = edit_dist / float(max(line1.N, line2.N))
		return edit_dist <= 1 or norm <= self.EDIT_DIST_THRESH

	def suffix_match(self, suffix, complete):
		'''
		A Suffix match is defined to be two lines whose end positions are within
			the distance threshold, the average character width is within
			a threshold and the edit distance between the suffix and the truncated
			other string is below a threshold.
		:param suffix: TextLine that might be the suffix of complete
		:param complete: TextLine that might contain suffix
		:return: bool Whether suffix is a suffix of complete or not
		'''
		if complete.N <= suffix.N:
			return False

		if utils.e_dist(complete.end_pos(), suffix.end_pos()) > self.dist_thresh:
			return False

		if not utils.ratio(complete.char_width(), suffix.char_width()) > self.SIZE_RATIO:
			return False

		edit_dist = Levenshtein.dist(complete.text[-1*suffix.N:], suffix.text)
		norm = edit_dist / float(suffix.N)
		return edit_dist <= 1 or norm <= self.EDIT_DIST_THRESH

	def prefix_match(self, prefix, complete):
		'''
		Same as suffix match, except with a prefix
		'''
		if complete.N <= prefix.N:
			return False

		if utils.e_dist(complete.pos, prefix.pos) > self.dist_thresh:
			return False

		if not utils.ratio(complete.char_width(), prefix.char_width()) > self.SIZE_RATIO:
			return False

		edit_dist = Levenshtein.dist(complete.text[:prefix.N], prefix.text)
		norm = edit_dist / float(prefix.N)
		return edit_dist <= 1 or norm <= self.EDIT_DIST_THRESH

	def _find_partial_matches():
		'''
		Finds Prefix/Suffix matches among the unmatched lines
		'''
		partial_matches = list()
		for line1 in self.lines1:
			if line1.matched:
				continue
			for line2 in self.lines2:
				if line2.matched:
					continue
				if self.prefix_match(line1, line2):
					partial_matches.append( (self.PREFIX1, line1, line2) )
				if self.prefix_match(line2, line1):
					partial_matches.append( (self.PREFIX2, line1, line2) )
				if self.suffix_match(line1, line2):
					partial_matches.append( (self.SUFFIX1, line1, line2) )
				if self.suffix_match(line2, line1):
					partial_matches.append( (self.SUFFIX2, line1, line2) )
		return parital_matches

	def _condense_matches(self, partials):
		condensed = list()
		# complete lines to the partials in the other list
		one_two = collections.defaultdict(list)
		two_one = collections.defaultdict(list)
		for part in partials:
			op, line1, line2 = part
			if op in [self.PREFIX2, self.SUFFIX2]:
				one_two[line1].append(op, line2)
			if op in [self.PREFIX1, self.SUFFIX1]:
				two_one[line2].append(op, line1)

		# TODO: case of multiple prefixes or suffixes better
		for line1, matches in one_two.iteritems():
			prefix = suffix = None
			for match in matches:
				op, line = match
				if op == self.PREFIX2:
					prefix = line
				if op == self.SUFFIX2:
					suffix = line
			if prefix is None or suffix is None:
				continue
			l = prefix.N + suffix.N
			if utils.ratio(l, line1.N) > (1 - self.EDIT_DIST_THRESH):
				# we have an actual partial match
				condensed.append( (self.PARTIAL1, line1, prefix, suffix) )

		for line2, matches in two_one.iteritems():
			prefix = suffix = None
			for match in matches:
				op, line = match
				if op == self.PREFIX1:
					prefix = line
				if op == self.SUFFIX1:
					suffix = line
			if prefix is None or suffix is None:
				continue
			l = prefix.N + suffix.N
			if utils.ratio(l, line2.N) > (1 - self.EDIT_DIST_THRESH):
				# we have an actual partial match
				condensed.append( (self.PARTIAL2, line2, prefix, suffix) )
		return condensed

	def get_matches(self):
		'''
		:return: list of tuples 
		'''
		matches = self._find_perfect_matches()
		if self.partials:
			paritals = self._find_partial_matches()
			condensed = self._condense_matches(partials)
			# mark the matches
			for match in condensed:
				for line in match[1:]:
					line.matched = True
		return matches

	def print_matches(self, matches):
		print
		print "** Text Line Matches **"
		for match in matches:
			op = match[0]
			print "\t%s" % self.get_op(op)
			for line in match[1:]:
				print "\t%s" % str(line)
		print
		
	def merge(self):
		'''
		:return: list of TextLine - lines1, lines2 merged into one list
		'''
		pass











