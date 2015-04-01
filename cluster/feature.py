
import utils
import text
import lines
import components
import scipy.spatial
from constants import *


class FeatureSet(object):
	
	def __init__(self, width, height, rows, cols):
		self.width = width
		self.height = height
		self.rows = rows
		self.cols = cols
		self.size = (self.width, self.height)

	def name(self):
		return "uknw"

	def display():
		pass

	def copy(self):
		pass

	def global_sim(self, other):
		pass

	def region_sim(self, other):
		pass

	def region_sim_with_weights(self, other):
		pass

	def global_region_sim(self, other):
		pass

	def region_weights(self):
		pass

	def prune(self):
		pass

	def prune_final(self):
		pass

	def draw(self, draw):
		pass

	def aggregate(self, other):
		pass

	def push_away(self, other):
		pass

	def match_vector(self, other):
		pass


class LineFeatureSet(FeatureSet):
	
	def __init__(self, *args):
		super(LineFeatureSet, self).__init__(*args)
		self.lines = list()

	def display(self):
		print "%d total lines" % len(self.lines)
		for line in self.lines:
			print line

	def global_sim(self, other):
		matcher = self._get_matcher(other)
		return matcher.similarity()

	def region_sim(self, other):
		matcher = self._get_matcher(other)
		return matcher.similarity_by_region(self.rows, self.cols, self.size)[0]

	def region_sim_with_weights(self, other):
		matcher = self._get_matcher(other)
		return matcher.similarity_by_region(self.rows, self.cols, self.size)

	def global_region_sim(self, other):
		matcher = self._get_matcher(other)
		global_sim = matcher.similarity()
		region_sims = matcher.similarity_by_region(self.rows, self.cols, self.size)[0]
		sims = utils.flatten(region_sims)
		sims.insert(0, global_sim)
		return sims

	def _get_decay(self):
		return 0

	def prune(self):
		if DECAY:
			self._prune(self._get_decay(), 0)

	def prune_final(self):
		if self.lines:
			final_prune_thresh = max(map(lambda line: line.count, self.lines)) / FINAL_PRUNE_DIV
			self._prune(self._get_decay(), final_prune_thresh)

	def _prune(self, amount, thresh):
		map(lambda line: line.decay(amount), self.lines)
		self.lines = filter(lambda line: line.count > thresh, self.lines)

	def aggregate(self, other):
		matcher = self._get_matcher(other)
		self.lines = matcher.merge()

	def match_vector(self, other):
		matcher = self._get_matcher(other)
		return matcher.get_match_vector()

	def push_away(self, other):
		matcher = self._get_matcher(other)
		matcher.push_away(PUSH_AWAY_PERC)

		
class TextLineFeatureSet(LineFeatureSet):
	
	def __init__(self, width, height, rows, cols, f=None):
		super(TextLineFeatureSet, self).__init__(width, height, rows, cols)
		if f:
			line = f.readline().strip()
			while line:
				tokens = line.split()
				text = " ".join(tokens[4:])
				pos = ( int(tokens[0]), int(tokens[1]) )
				size = ( int(tokens[2]), int(tokens[3]) )
				self.lines.append(components.TextLine(text, pos, size))
				line = f.readline().strip()

	def name(self):
		return "text"

	def copy(self):
		new = TextLineFeatureSet(self.width, self.height, self.rows, self.cols)
		new.lines = map(lambda line: line.copy(), self.lines)
		return new

	def region_weights(self):
		thresh = TEXT_THRESH_MULT * max(self.width, self.height)
		matcher = text.TextLineMatcher(self.lines, list(), thresh, PARTIAL_TEXT_MATCHES)
		return matcher.similarity_by_region(self.rows, self.cols, (self.width, self.height) )[1]

	def _get_matcher(self, other):
		thresh = TEXT_THRESH_MULT * max(self.size)
		matcher = text.TextLineMatcher(self.lines, other.lines, thresh, PARTIAL_TEXT_MATCHES)
		return matcher

	def draw(self, draw):
		for line in self.lines:
			#fill = colors[idx % len(colors)] if colortext else "black"
			fill = "black"
			draw.text(line.pos, line.text, font=utils.get_font(line.text, line.size[0]), fill=fill)
			#draw.text( line.pos, "%.2f" % line.count, fill=TEXT_COUNT_COLOR)

	def _get_decay(self):
		return TEXT_DECAY 

class TextLineKDTree(LineFeatureSet):
	
	def __init__(self, width, height, rows, cols, f=None):
		super(TextLineKDTree, self).__init__(width, height, rows, cols)
		if f:
			line = f.readline().strip()
			while line:
				tokens = line.split()
				text = " ".join(tokens[4:])
				pos = ( int(tokens[0]), int(tokens[1]) )
				size = ( int(tokens[2]), int(tokens[3]) )
				self.lines.append(components.TextLine(text, pos, size))
				line = f.readline().strip()
			self.kd_tree = self.form_kd_tree(self.lines)

	def form_kd_tree(self, lines):
		locations = map(lambda line: line.pos, lines)
		kd_tree = scipy.spatial.KDTree(locations)
		return kd_tree

	def name(self):
		return "text"

	def copy(self):
		new = TextLineKDTree(self.width, self.height, self.rows, self.cols)
		new.lines = map(lambda line: line.copy(), self.lines)
		new.kd_tree = new.form_kd_tree(new.lines)
		return new

	def region_weights(self):
		thresh = TEXT_THRESH_MULT * max(self.width, self.height)
		matcher = text.TextLineKDMatcher(self, self, thresh, PARTIAL_TEXT_MATCHES)
		return matcher.similarity_by_region(self.rows, self.cols, (self.width, self.height) )[1]

	def _get_matcher(self, other):
		thresh = TEXT_THRESH_MULT * max(self.size)
		matcher = text.TextLineKDMatcher(self, other, thresh, PARTIAL_TEXT_MATCHES)
		return matcher

	def draw(self, draw):
		for line in self.lines:
			#fill = colors[idx % len(colors)] if colortext else "black"
			fill = "black"
			draw.text(line.pos, line.text, font=utils.get_font(line.text, line.size[0]), fill=fill)
			draw.text( line.pos, "%.2f" % line.count, fill=TEXT_COUNT_COLOR)

	def _get_decay(self):
		return TEXT_DECAY 

	def _prune(self, amount, thresh):
		map(lambda line: line.decay(amount), self.lines)
		self.lines = filter(lambda line: line.count > thresh, self.lines)
		self.kd_tree = self.form_kd_tree(self.lines)

	def aggregate(self, other):
		matcher = self._get_matcher(other)
		self.lines = matcher.merge()
		#self.kd_tree = self.form_kd_tree(self.lines)

class GridLineFeatureSet(LineFeatureSet):

	def __init__(self, width, height, rows, cols, orien, f=None):
		super(GridLineFeatureSet, self).__init__(width, height, rows, cols)
		self.lines = list()
		self.orien = orien

		if f:
			line = f.readline().strip()
			while line:
				tokens = line.split()
				pos = ( int(tokens[0]), int(tokens[1]) )
				length = int(tokens[2])
				thick = int(tokens[3])
				self.lines.append(components.Line(orien, pos, length, thick))
				line = f.readline().strip()

	def name(self):
		return "horz" if self.orien == components.Line.HORIZONTAL else "vert"

	def region_weights(self):
		thresh_dist = LINE_THRESH_MULT * max(self.width, self.height)
		matcher = lines.LMatcher(self.lines, list(), thresh_dist, self.size)
		return matcher.similarity_by_region(self.rows, self.cols, (self.width, self.height) )[1]

	def copy(self):
		new = GridLineFeatureSet(self.width, self.height, self.rows, self.cols, self.orien)
		new.lines = map(lambda line: line.copy(), self.lines)
		return new

	def _get_matcher(self, other):
		thresh_dist = LINE_THRESH_MULT * max(self.width, self.height)
		matcher = lines.LMatcher(self.lines, other.lines, thresh_dist, self.size)
		return matcher

	def draw(self, draw):
		for line in self.lines:
			if line.is_horizontal():
				draw.line( (utils.tup_int(line.pos), utils.tup_int( (line.pos[0] + line.length, line.pos[1]) )) ,
							width=int(line.thickness * 2), fill=HORZ_COLOR)
			else:
				draw.line( (utils.tup_int(line.pos), utils.tup_int( (line.pos[0], line.pos[1] + line.length) )) ,
							width=int(line.thickness * 2), fill=VERT_COLOR)
			#draw.text( utils.tup_int(line.pos), "%.2f" % line.count, fill=GRID_LINE_COUNT_COLOR)

	def _get_decay(self):
		return LINE_DECAY 

class SurfFeatureSet(FeatureSet):

	def __init__(self, width, height, rows, cols, f=None):
		super(SurfFeatureSet, self).__init__(width, height, rows, cols)
		self.cell_width = int(self.width / float(self.cols) + 1)
		self.cell_height = int(self.height / float(self.rows) + 1)

		if f:
			self.codebook_size = int(f.readline().strip())

			self.global_histogram = [0] * self.codebook_size
			self.global_norm = 0.0
			self.region_histograms = dict()
			self.region_norms = dict()
			for r in xrange(self.rows):
				for c in xrange(self.cols):
					self.region_histograms[r,c] = [0] * self.codebook_size
					self.region_norms[r,c] = 0.0
			line = f.readline().strip()
			while line:
				tokens = line.split()
				x = int(tokens[0])
				y = int(tokens[1])
				code = int(tokens[2])

				self.global_histogram[code] += 1
				self.global_norm += 1

				region = self._get_region(x, y)
				self.region_histograms[region][code] += 1
				self.region_norms[region] += 1

				line = f.readline().strip()
			self._form_distributions()

	def display(self):
		print "global: %d %s" % (self.global_norm, " ".join(map(lambda x: "%0.1f" % (100 * x), self.global_distribution)))
		for r in xrange(self.rows):
			for c in xrange(self.cols):
				print "region %s: %d %s" % ( (r, c), self.region_norms[r,c], " ".join(map(lambda x: "%.3f" % x, self.region_distributions[r,c])))

	def name(self):
		return "surf"

	def copy(self):
		new = SurfFeatureSet(self.width, self.height, self.rows, self.cols)
		new.cell_width = self.cell_width
		new.cell_height = self.cell_height
		new.codebook_size = self.codebook_size
		new.global_norm = self.global_norm
		new.global_histogram = self.global_histogram[:]
		new.region_norms = self.region_norms.copy()
		new.region_histograms = {region: self.region_histograms[region][:] for region in self.region_histograms}
		new._form_distributions()
		return new


	def _form_distributions(self):
		self.global_distribution = self._norm_histo(self.global_histogram, self.global_norm)
		self.region_distributions = {region: self._norm_histo(self.region_histograms[region], self.region_norms[region]) 
										for region in self.region_histograms}

	def _get_region(self, x, y):
		col = min(x / self.cell_width, self.cols - 1)
		row = min(y / self.cell_height, self.rows - 1)
		return (row, col)
	
	def _dict_to_list(self, d):
		l = list()
		for r in xrange(self.rows):
			for c in xrange(self.cols):
				l.append(d[r,c])
		return l

	def _dict_to_mat(self, d):
		m = list()
		for r in xrange(self.rows):
			row = list()
			for c in xrange(self.cols):
				row.append(d[r,c])
			m.append(row)
		return m

	def _norm_histo(self, histo, norm):
		return map(lambda x: x / norm if norm else 0, histo)

	def global_sim(self, other):
		return utils.bhattacharyya_coeff(self.global_distribution, other.global_distribution)

	def region_sim(self, other):
		region_sims = {region: utils.bhattacharyya_coeff(self.region_distributions[region], other.region_distributions[region])
						for region in self.region_histograms}
		return self._dict_to_mat(region_sims)

	def region_sim_with_weights(self, other):
		region_sims = self.region_sim(other)
		weights = self.region_weights()
		return region_sims, weights

	def global_region_sim(self, other):
		global_sim = self.global_sim(other)
		region_sims = self.region_sim(other)
		sims = utils.flatten(region_sims)
		sims.insert(0, global_sim)
		return sims

	def region_weights(self):
		weights = {region: self.region_norms[region] / self.global_norm for region in self.region_histograms}
		return self._dict_to_mat(weights)

	def aggregate(self, other):
		self.global_norm += other.global_norm
		for x in xrange(self.codebook_size):
			self.global_histogram[x] += other.global_histogram[x]
		for r in xrange(self.rows):
			for c in xrange(self.cols):
				for x in xrange(self.codebook_size):
					self.region_histograms[r,c][x] += other.region_histograms[r,c][x]
				self.region_norms[r,c] += other.region_norms[r,c]
		self._form_distributions()
	
