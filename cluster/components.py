
import collections
import dictionary
import utils
import math
import Levenshtein

class Feature(object):
    
	def __init__(self, count=1, weight=1.0):
		'''
		:param count: Number of times such feature occurs
		:param weight: [0-1.0] measure of the significance of a feature
		'''
		self.count = count
		self.weight = weight
		self.matched = False

	def match_value(self):
		'''
		In template matching, how much does this feature actually matter?
		'''
		return self.count * self.weight

	def decay(self, amount):
		'''
		:param amount: num - decrements the count by amount
		'''
		self.count -= amount
			

class Line(Feature):
	# These are line segments

	# important that these don't change
	VERTICAL = 0
	HORIZONTAL = 1
	
	def __init__(self, orientation, pos, length, thickness):
		'''
		:param orientation: Line.HORIZONTAL or Line.VERTICAL
		:param position: (x, y) coordinate of lowest endpoint (along center of line)
		:param length: int length of the line in pixels
		:param thickness: int line thickness
		'''
		Feature.__init__(self)
		self.orien = orientation
		self.pos = pos
		self.length = length
		self.thickness = thickness

	def copy(self):
		cpy = Line(self.orien, self.pos, self.length, self.thickness)
		cpy.count = self.count
		cpy.weight = self.weight
		return cpy

	def __str__(self):
		return "%s pos: %s length: %d thickness %d count %.2f id: %s" % \
				 ('H' if self.orien else 'V', self.pos, self.length, self.thickness, self.count, id(self))

	def is_horizontal(self):
		return self.orien == Line.HORIZONTAL

	def is_vertical(self):
		return self.orien == Line.VERTICAL
		
	def match_value(self):
		'''
		Overides Feature.match_value
		Gives lines value proportional to log(length)
		'''
		return Feature.match_value(self) * math.log(self.length)

	def truncate(self, size):
		'''
		:param size: (width, height)
		Truncates the endpoints of the line to be within the rectangle defined by size
		'''
		if self.pos[0] < 0:
			self.pos = (0, self.pos[1])
		if self.pos[1] < 0:
			self.pos = (self.pos[0], 0)
		if self.is_horizontal() and (self.pos[0] + self.length) >= size[0]:
			self.length = size[0] - self.pos[0] - 1
			#print "Truncating: ", self
		elif self.is_vertical() and (self.pos[1] + self.length) >= size[1]:
			self.length = size[1] - self.pos[1] - 1
			#print "Truncating: ", self

	def end_pos(self):
		if self.orien == Line.HORIZONTAL:
			return (self.pos[0], self.pos[1] + self.length)
		else:
			return (self.pos[0] + self.length, self.pos[1])

	def length_range(self, offset=0):
		return (self.pos[1-self.orien] + offset, self.pos[1-self.orien] + self.length + offset)
		

class TextLine(Feature):

	NO_MATCH = 0
	COMPLETE_MATCH = 1
	PREFIX_MATCH = 2
	SUFFIX_MATCH = 3

	MATCHING_THRESH = 0.20
	
	def __init__(self, chars, position, size):
		'''
		:param text: str, contents of the text line (length N) from OCR
		:param confidences: list(float) confidence measure of each character
		:param position: (x, y) coordinate of the upper left corner of the bounding rectangle
		:param size: (x, y) width and height of the bounding rectangle
		'''
		Feature.__init__(self)
		self.chars = chars
		self.set_text()
		self.pos = position
		self.size = size
		#self.width = size[0]
		#self.height = size[1]
		self.N = len(self.text)
		self.members = collections.Counter()

	def find_median(self):
		most_common = self.members.most_common(2);

		# we have a tie for the mode
		if most_common[0][1] == most_common[0][0]:
			# account for prefix/suffix stuff too
			#text = map(lambda ele: ele if isinstance(ele, str) else ele[0] + ele[1], self.members)
			#print "\t", text
			return Levenshtein.median(self.members.keys(), self.members.values())
		else:
			return most_common[0][0]

	def aggregate(self, other):
		if not self.members:
			self.members[self.text] += 1
		self.members[other.text] += 1
		weights = [self.count, other.count]
		x = utils.wavg([self.pos[0], other.pos[0]], weights)
		y = utils.wavg([self.pos[1], other.pos[1]], weights)
		w = utils.wavg([self.size[0], other.size[0]], weights)
		h = utils.wavg([self.size[1], other.size[1]], weights)
		self.pos = (x, y)
		self.size = (w, h)
		self.text = self.find_median()
		self.N = len(self.text)
		self.count += other.count
		# because we are taking the median string, the chars no longer matter
		if hasattr(self, "chars"):
			del self.chars

	def aggregate_partial(self, prefix, suffix):
		''' aggregate self with a prefix and a suffix text lines '''
		if not self.members:
			self.members.append(self.text)
		#self.members.append( (prefix.text, suffix.text) )
		self.members[prefix.text + suffix.text] += 1
		weights = [self.count, (prefix.count + suffix.count) / 2]
		x = utils.wavg([self.pos[0], prefix.pos[0]], weights)
		y = utils.wavg([self.pos[1], prefix.pos[1]], weights)
		w = utils.wavg([self.size[0], prefix.size[0] + suffix.size[0]], weights)
		h = utils.wavg([self.size[1], (prefix.size[1] + suffix.size[1]) / 2], weights)
		self.pos = (x, y)
		self.size = (w, h)
		self.text = self.find_median()
		self.N = len(self.text)
		self.count += weights[1]

	def aggregate_as_prefix(self, other):
		''' self is a prefix of other '''
		if not self.members:
			self.members.append(self.text)
		self.members.append(other.text[0:self.N])
		weights = [self.count, other.count / 2]
		x = utils.wavg([self.pos[0], other.pos[0]], weights)
		y = utils.wavg([self.pos[1], other.pos[1]], weights)
		w = utils.wavg([self.size[0], other.size[0] * (self.N / float(other.N)) ], weights)
		h = utils.wavg([self.size[1], other.size[1]], weights)
		self.pos = (x, y)
		self.size = (w, h)
		self.text = self.find_median()
		self.N = len(self.text)
		self.count += weights[1]

	def aggregate_as_suffix(self, other):
		''' self is a suffix of other '''
		if not self.members:
			self.members.append(self.text)
		self.members.append(other.text[-1 * self.N:])
		weights = [self.count, other.count / 2]
		x = utils.wavg([self.pos[0], other.pos[0] + (other.size[0] * (self.N / float(other.N)))], weights)
		y = utils.wavg([self.pos[1], other.pos[1]], weights)
		w = utils.wavg([self.size[0], other.size[0] * self.N / float(other.N) ], weights)
		h = utils.wavg([self.size[1], other.size[1]], weights)
		self.pos = (x, y)
		self.size = (w, h)
		self.text = self.find_median()
		self.N = len(self.text)
		self.count += weights[1]

	def copy(self):
		chars_copy = list()
		for char in self.chars:
			chars_copy.append(char.copy())
		cpy = TextLine(chars_copy, self.pos, self.size)
		cpy.count = self.count
		cpy.weight = self.weight
		cpy.members = collections.Counter(self.members)
		#cpy.members = list(self.members)
		return cpy

	def match_value(self):
		'''
		Overides Feature.match_value
		Gives text lines value proportional to their length
		'''
		return Feature.match_value(self) * self.N

	def set_text(self):
		if self.chars:
			ul = self.pos = self.chars[0].pos
			br = self.chars[-1].pos2
			self.size = (br[0] - ul[0], br[1] - ul[1])
		self.text = "".join(map(lambda char: char.val, self.chars))
		self.N = len(self.text)
		
	def __str__(self):
		return "text: %r pos: %s size: %s" % (self.text, self.pos, self.size)

	def __repr__(self):
		return self.__str__()
		
	def filter_nonalpha(self):
		for x in xrange(self.N):
			c = self.chars[x].val
			if ord(c) > 127 or not (c.isalpha() or c.isspace()):
				self.chars[x].val = ' '
		self.set_text()

	def has_dict_word(self, min_len=3):
		for word in self.text.split():
			if dictionary.is_word(word) and len(word) >= min_len:
				return True
		return False

	def trim(self):
		while self.chars and self.chars[0].val.isspace():
			self.chars.pop(0)
		while self.chars and self.chars[-1].val.isspace():
			self.chars.pop(-1)
		self.set_text()

	def condense_space(self):
		new_chars = []
		prev_space = False
		for x in xrange(self.N):
			if not (prev_space and self.chars[x].val.isspace()):
				new_chars.append(self.chars[x])
			prev_space = self.chars[x].val.isspace()
		self.chars = new_chars
		self.set_text()

	def end_pos(self):
		return (self.pos[0] + self.size[0], self.pos[1])

	def bottom_right(self):
		return (self.pos[0] + self.size[0], self.pos[1] + self.size[1])

	def char_width(self):
		return float(self.size[0]) / self.N

	def matches(self, other, dist_thresh):
		if utils.e_dist(self.pos, other.pos) > dist_thresh:
			return TextLine.NO_MATCH
		if utils.close_match(self.text, other.text, TextLine.MATCHING_THRESH):
			return TextLine.COMPLETE_MATCH
		if other.text.startswith(self.text):
			# self is a prefix of other
			return TextLine.PREFIX_MATCH
		if other.text.endswith(self.text):
			# self is a suffix of other
			return TextLine.SUFFIX_MATCH
		return TextLine.NO_MATCH


class Char:

	bool_labels = ['wordStart', 'wordFromDictionary', 'wordNormal',
						'wordNumeric', 'wordIdentifier', 'wordPenalty']
	int_labels = ['charConfidence', 'serifProbability', 'meanStrokeWidth']
	
	#def __init__(self, value, d):
	def __init__(self, value, ul, br):
		self.val = value
		#self.attributes = d.copy()
		#l = d['l']
		#t = d['t']
		#r = d['r']
		#b = d['b']
		#self.pos = (l, t)
		self.pos = ul
		self.pos2 = br

		# Not currently used for anything
		#self.size = (r - l, b - t)
		#self.area = self.size[0] * self.size[1]
		#for bool_label in Char.bool_labels:
		#	if bool_label in self.attributes:
		#		self.attributes[bool_label] = bool(self.attributes[bool_label])
		#	else:
		#		self.attributes[bool_label] = False

	def copy(self):
		#return Char(self.val, self.attributes)
		return Char(self.val, self.pos, self.pos2)

	#def get_attr(self, attr):
	#	return self.attributes.get(attr)
	#
	#def set_attr(self, attr, val):
	#	self.attributes[attr] = val
		
