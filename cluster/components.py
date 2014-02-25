
import dictionary
import utils

class Feature(object):
    
	def __init__(self, count=1, weight=1.0):
		'''
		:param count: Number of times such feature occurs
		:param weight: [0-1.0] measure of the significance of a feature
		'''
		self.count = count
		self.weight = weight

	def match_value(self):
		'''
		In template matching, how much does this feature actually matter?
		'''
		return self.count * self.weight
        

class Line(Feature):
	# These are line segments
	
	def __init__(self, position, size):
		'''
		:param position: (x, y) coordinate of the upper left corner of the bounding rectangle
		:param size: (x, y) width and height of the bounding rectangle
		'''
		Feature.__init__(self)
		self.position = position
		self.size = size
		self.width = size[0]
		self.height = size[1]

	def is_horizontal(self):
		return self.width > self.height

	def is_vertical(self):
		return not self.is_horizontal
		

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

	def copy(self):
		chars_copy = list()
		for char in self.chars:
			chars_copy.append(char.copy())
		cpy = TextLine(chars_copy, self.pos, self.size)
		return cpy

	def match_value(self):
		'''
		Overides Feature.match_value
		Gives text lines value proportional to their length
		'''
		return Feature.match_value(self) * self.N

	def set_text(self):
		# TODO: why doesn't this update the position.  Maybe chars have been removed?
		self.text = "".join(map(lambda char: char.val, self.chars))
		self.N = len(self.text)
		
	def __str__(self):
		return "text: %r pos: %s size: %s" % (self.text, self.pos, self.size)

	def __repr__(self):
		return self.__str__()
		
	def filter_nonalpha(self):
		for x in xrange(self.N):
			c = self.chars[x].val
			if not (c.isalpha() or c.isspace()):
				self.chars[x].val = ' '
		#self.chars = filter(lambda char: char.val.isalpha() or char.val.isspace(), self.chars)
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
		