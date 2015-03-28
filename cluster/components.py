
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
		return "%s pos: %s length: %d thickness %d count %.2f" % \
				 ('H' if self.orien else 'V', self.pos, self.length, self.thickness, self.count)

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
		

	#def end_pos(self):
	#	if self.orien == Line.HORIZONTAL:
	#		return (self.pos[0], self.pos[1] + self.length)
	#	else:
	#		return (self.pos[0] + self.length, self.pos[1])

	def length_range(self, offset=0):
		return (self.pos[1-self.orien] + offset, self.pos[1-self.orien] + self.length + offset)
		

class TextLine(Feature):

	NO_MATCH = 0
	COMPLETE_MATCH = 1
	PREFIX_MATCH = 2
	SUFFIX_MATCH = 3

	MATCHING_THRESH = 0.20
	
	def __init__(self, text, position, size):
		'''
		:param text: str, contents of the text line (length N) from OCR
		:param confidences: list(float) confidence measure of each character
		:param position: (x, y) coordinate of the upper left corner of the bounding rectangle
		:param size: (x, y) width and height of the bounding rectangle
		'''
		Feature.__init__(self)
		self.text = text
		self.pos = position
		self.size = size
		self.N = len(self.text)
		self.members = collections.Counter()
		self.members[self.text] += 1
		self.set_end_pos()

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
		self.members[other.text] += 1
		if self.count + other.count == 0:
			self.count = 1
			other.count = 1
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
		self.set_end_pos()

	def aggregate_partial(self, prefix, suffix):
		''' aggregate self with a prefix and a suffix text lines '''
		self.members[prefix.text + suffix.text] += 1
		weights = [self.count, (prefix.count + suffix.count) / 2.0]
		x = utils.wavg([self.pos[0], prefix.pos[0]], weights)
		y = utils.wavg([self.pos[1], prefix.pos[1]], weights)
		w = utils.wavg([self.size[0], prefix.size[0] + suffix.size[0]], weights)
		h = utils.wavg([self.size[1], (prefix.size[1] + suffix.size[1]) / 2.0], weights)
		self.pos = (x, y)
		self.size = (w, h)
		self.text = self.find_median()
		self.N = len(self.text)
		self.count += weights[1]
		self.set_end_pos()

	def aggregate_as_prefix(self, other):
		''' self is a prefix of other '''
		self.members[other.text[0:self.N]] += 1
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
		self.set_end_pos()

	def aggregate_as_suffix(self, other):
		''' self is a suffix of other '''
		self.members[other.text[-1 * self.N:]] += 1
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
		self.set_end_pos()

	def copy(self):
		cpy = TextLine(self.text, self.pos, self.size)
		cpy.count = self.count
		cpy.weight = self.weight
		cpy.members = collections.Counter(self.members)
		cpy.set_end_pos()
		return cpy

	def match_value(self):
		'''
		Overides Feature.match_value
		Gives text lines value proportional to their length
		'''
		return Feature.match_value(self) * self.N

	def __str__(self):
		return "text: %r pos: %s size: %s" % (self.text, self.pos, self.size)

	def __repr__(self):
		return self.__str__()
		
	def set_end_pos(self):
		self.end_pos = (self.pos[0] + self.size[0], self.pos[1])

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

