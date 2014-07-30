
import Image
import ImageDraw
import ImageFont
import json
import string
import utils
import os

import components
import profiles
import lines
import text
import ocr

_file_extensions = [".jpg", ".xml", "_line.xml", "_FormType.txt", "_endpoints.xml"]

DECAY = True

_line_thresh_mult = 0.05
_line_decay_amount = 1.0 / 10 if DECAY else 0
_text_decay_amount = 1.0 / 15 if DECAY else 0

def get_doc(_dir, basename):
	paths = map(lambda ext: os.path.join(_dir, basename + ext), _file_extensions)

	document = Document(basename, paths)
	return document

def get_docs(_dir, pr=True):
	docs = []
	num_loaded = 0
	num_exceptions = 0
	if pr:
		print "Loading Docs from:", _dir
		basenames = set()
		image_names = os.listdir(_dir)
		for name in image_names:
			if name.endswith(".jpg"):
				basename = os.path.splitext(name)[0]
				basenames.add(basename)
		for basename in basenames:
			try:
				docs.append(get_doc(_dir, basename))
				num_loaded += 1
				if pr and num_loaded % 10 == 0:
					print "\tLoaded %d documents" % num_loaded
			except:
				num_exceptions += 1
	if pr:
		print "\t%d Docs read" % num_loaded
		if num_exceptions:
			print "\t%d Docs could not be read" % num_exceptions
	return docs, num_exceptions

	

def get_docs_nested(data_dir, pr=True):
	all_docs = []
	total_exceptions = 0
	for _dir in os.listdir(data_dir):
		r_dir = os.path.join(data_dir, _dir)
		docs, num_exceptions = get_docs(r_dir, pr)
		all_docs += docs
		total_exceptions += num_exceptions
	if pr:
		print "Finished loading all documents"
		print "%d Total docs read" % len(all_docs)
		if total_exceptions:
			print "%d Toal docs could not be read" % total_exceptions
	return all_docs




LAZY = True
ALLOW_PARTIAL_MATCHES = False
DEBUG = False

class Document:
	
	def __init__(self, _id, paths, original=True):
		self._id = _id
		self.paths = paths
		self.image_path = paths[0]
		self.ocr_path = paths[1]
		self.prof_path = paths[2] # path of projection profiles
		self.form_path = paths[3]
		self.endpoints_path = paths[4]

		self.loaded = False
		if not LAZY and original:
			self.load()

	def copy(self, new_id):
		self._load_check()
		cpy = Document(new_id, self.paths, original=False)
		cpy.loaded = True  # makes sure we never load data from files

		cpy.label = self.label
		cpy.text_lines = map(lambda line: line.copy(), self.text_lines)
		cpy.h_lines = map(lambda line: line.copy(), self.h_lines)
		cpy.v_lines = map(lambda line: line.copy(), self.v_lines)
		cpy.size = self.size

		cpy.char_mass = self.char_mass
		#cpy.h_line_mass = self.h_line_mass
		#cpy.v_line_mass = self.v_line_mass

		return cpy

	def display(self):
		self._load_check()
		print "Document %s" % self._id
		for line in self.textLines:
			print line
		print

	def _load_check(self):
		if not self.loaded:
			self.load()

	def load(self):
		#image = Image.open(image_path)
		#del image

		self.label = filter(lambda x: x in string.printable, open(self.form_path).read())
		#print repr(self.label)

		# heavy operations
		self.text_lines = ocr.clean_lines(ocr.extract_text_lines(self.ocr_path))
		self._set_total_char_mass()

		self.h_lines, self.v_lines = lines.read_lines(self.endpoints_path)
		assert self.h_lines
		assert self.v_lines

		#profs = profiles.extract_profiles(self.prof_path)
		#self.horz_prof = profs['HorizontalLineProfile']
		#self.vert_prof = profs['VerticalLineProfile']
		#self.horz_prof = profs['HorizontalLineProfile']
		#self.vert_prof = profs['VerticalLineProfile']
		#self.size = (len(self.vert_prof), len(self.horz_prof))

		# TODO: finish this
		self.size = profiles.get_size(self.prof_path)

		# inefficient for now
		#profs = profiles.extract_profiles(self.prof_path)
		#self.size = (len(profs['VerticalLineProfile']), len(profs['HorizontalLineProfile']))

		self.loaded = True

	def _set_total_char_mass(self):
		self.char_mass = sum(map(lambda line: line.match_value(), self.text_lines))

	def _get_matching_char_mass(self):
		mass = 0.0
		for line in self.text_lines:
			if line.matched:
				mass += line.match_value()
		return mass

	def _get_char_mass_ratio(self):
		return self._get_matching_char_mass() / self.char_mass if self.char_mass else 0.0

	def _find_matching_text_line(self, query_line, thresh_dist):
		for line in self.text_lines:
			if line.matched:
				continue
			match = line.matches(query_line, thresh_dist)
			if match:
				if ALLOW_PARTIAL_MATCHES and match != components.TextLine.COMPLETE_MATCH:
					if match == components.TextLine.PREFIX_MATCH:
						# line is a prefix of query_line
						chars = query_line.chars[cmp_line.N:]
					else:  
						# line is a suffix of query_line
						chars = query_line.chars[0:cmp_line.N]
					first_char = chars[0]
					last_char = chars[-1]
					pos1 = first_char.pos
					pos2 = last_char.pos
					size = (pos2[0] - pos1[0], line.size[1])  # not that size is being used anywhere...

					# create dummy TextLine to match the rest of query_line
					remaining_line = components.TextLine(chars, pos1, size)
					remaining_line.matched = False
					if DEBUG:
						print "partial match:"
						print "\t", query_line
						print "\t", line
						print "\t", "Searching for remaining:", remaining_line

					# recurse
					partial_match = self._find_matching_text_line(remaning_line, thresh_dist)
					if not partial_match:
						return None

				line.matched = True
				query_line.matched = True
				if DEBUG:
					print line
				return line

		return None

	def similarity(self, other):
		return self.global_similarity(other)

	def global_similarity(self, other):
		sims = self.similarities_by_name(other).values()
		return utils.harmonic_mean_list(sims)
                    
	def similarities_by_name(self, other):
		sims = dict()
		funs = self.similarity_functions()
		for fun in funs:
			sims[fun] = funs[fun](other)
		return sims

	def similarity_functions(self):
		funs = dict()
		funs['text_line'] = (lambda other: self.text_line_similarity(other))
		funs['h_line'] = (lambda other: self.h_line_similarity(other))
		funs['v_line'] = (lambda other: self.v_line_similarity(other))
		return funs

	def similarity_function_names(self):
		return ['text_line', 'h_line', 'v_line']

	def text_line_similarity(self, other):
		self._load_check()
		other._load_check()
		thresh_dist = 0.10 * max(max(self.size), max(other.size))  # % of largest dimension

		text_matcher = text.TextLineMatcher(self.text_lines, other.text_lines, thresh_dist, True)
		matches = text_matcher.get_matches()
		text_matcher.print_matches(matches)
		return text_matcher.similarity()

	def _text_line_similarity(self, other):
		self._load_check()
		other._load_check()
		thresh_dist = 0.10 * max(max(self.size), max(other.size))  # % of largest dimension

		self.clear_text_matches()
		other.clear_text_matches()

		# each matched line has a flag set by has_match indicating that it matches
		for line in other.text_lines:
			self._find_matching_text_line(line, thresh_dist)
		# we do it both ways to catch prefix/suffix matches in both directions
		if ALLOW_PARTIAL_MATCHES:
			for line in self.text_lines:
				other._find_matching_text_line(line, thresh_dist)

		my_ratio = self._get_char_mass_ratio()
		other_ratio = other._get_char_mass_ratio()
		return utils.harmonic_mean(my_ratio, other_ratio)

	def line_similarity(self, other):
		h_sim = self.h_line_similarity(other)
		v_sim = self.v_line_similarity(other)
		return utils.harmonic_mean(h_sim, v_sim)

	def h_line_similarity(self, other):
		self._load_check()
		other._load_check()

		h_thresh_dist = _line_thresh_mult * max(self.size[0], other.size[0]) 
		h_matcher = lines.LMatcher(self.h_lines, other.h_lines, h_thresh_dist)
		matches = h_matcher.get_matches()
		#print "Horizontal Matricies:"
		#h_matcher.display()
		#ops = h_matcher.get_operations()
		#h_matcher.print_ops(ops)

		return h_matcher.similarity()

	def v_line_similarity(self, other):
		self._load_check()
		other._load_check()
		#self.clear_v_line_matches()
		#other.clear_v_line_matches()

		v_thresh_dist = _line_thresh_mult * max(self.size[1], other.size[1]) 
		v_matcher = lines.LMatcher(self.v_lines, other.v_lines, v_thresh_dist)
		return v_matcher.similarity()
		#matches = v_matcher.get_matches()
		#print "Vertical"
		#v_matcher.display()

	def clear_text_matches(self):
		for line in self.text_lines:
			line.matched = False

	def _aggregate_text(self, other):
		thresh_dist = 0.10 * max(max(self.size), max(other.size))  # % of largest dimension

		self.clear_text_matches()
		other.clear_text_matches()

		to_add = list()
		for line in other.text_lines:
			matched_line = self._find_matching_text_line(line, thresh_dist)
			if matched_line:
				matched_line.aggregate(line)
				#x = (line.count * line.pos[0] + matched_line.count * matched_line.pos[0]
				#	) / (line.count + matched_line.count)
				#y = (line.count * line.pos[1] + matched_line.count * matched_line.pos[1]
				#	) / (line.count + matched_line.count)

				## Don't worry about size for now
				##width = (line.count * line.size[0] + matched_line.count * matched_line.size[0]) /
				##	(line.count + matched_line.count)
				##height = (line.count * line.size[1] + matched_line.count * matched_line.size[1]) /
				##	(line.count + matched_line.count)
				## hmmm, what if the text is not an exact match...
				## partial matches?

				## translations
				#dx = x - matched_line.pos[0]
				#dy = y - matched_line.pos[1]
				#for char in matched_line.chars:
				#	new_pos = (char.pos[0] + dx, char.pos[1] + dy)
				#	char.pos = new_pos
				#	char.pos2 = (char.pos2[0] + dx, char.pos2[1] + dy)
				#	#char.attributes['l'] += dx
				#	#char.attributes['r'] += dx
				#	#char.attributes['t'] += dy
				#	#char.attributes['b'] += dy

				#matched_line.count += line.count
				#matched_line.pos = (x, y)
				##matched_line.size = (width, height)
			else:
				to_add.append(line.copy())

		self.text_lines += to_add
		self._set_total_char_mass()

	def _aggregate_h_lines(self, other):
		h_thresh_dist = _line_thresh_mult * max(self.size[0], other.size[0]) 
		h_matcher = lines.LMatcher(self.h_lines, other.h_lines, h_thresh_dist)
		#ops = h_matcher.get_operations()
		#h_matcher.print_ops(ops)
		self.h_lines = h_matcher.get_merged_lines()

	def _aggregate_v_lines(self, other):
		v_thresh_dist = _line_thresh_mult * max(self.size[1], other.size[1]) 
		v_matcher = lines.LMatcher(self.v_lines, other.v_lines, v_thresh_dist)
		self.v_lines = v_matcher.get_merged_lines()
		
	def aggregate(self, other):
		self._load_check()
		other._load_check()

		self.size = (max(self.size[0], other.size[0]), max(self.size[1], other.size[1]))

		self._aggregate_text(other)
		self._aggregate_h_lines(other)
		self._aggregate_v_lines(other)

		self.prune()

	def prune(self):
		self._prune_text(0, _text_decay_amount)
		self._prune_h_lines(0, _line_decay_amount)
		self._prune_v_lines(0, _line_decay_amount)

	def final_prune(self):
		get_prune_val = lambda lines: max(map(lambda line: line.count, lines)) / 10.0
		if self.text_lines:
			self._prune_text(get_prune_val(self.text_lines), _text_decay_amount)
		if self.h_lines:
			self._prune_h_lines(get_prune_val(self.h_lines), _line_decay_amount)
		if self.v_lines:
			self._prune_v_lines(get_prune_val(self.v_lines), _line_decay_amount)

	def _prune_text(self, thresh, amount):
		map(lambda line: line.decay(amount), self.text_lines)
		self.text_lines = filter(lambda line: line.count > thresh, self.text_lines)

	def _prune_h_lines(self, thresh, amount):
		map(lambda line: line.decay(amount), self.h_lines)
		tmp = filter(lambda line: line.count > thresh, self.h_lines)
		if not tmp:
			print self.h_lines
		self.h_lines = tmp

	def _prune_v_lines(self, thresh, amount):
		map(lambda line: line.decay(amount), self.v_lines)
		tmp = filter(lambda line: line.count > thresh, self.v_lines)
		if not tmp:
			print self.v_lines
		self.v_lines = tmp

	def draw(self, colortext=False):
		'''
		:param colortext: True for drawing each text line a different color, False for all black
		:return Image:
		'''
		self._load_check()
		im = Image.new('RGB', self.size, 'white')
		draw = ImageDraw.Draw(im)
		colors = utils.colors
		idx = 0
		for line in self.h_lines:
			color = 'orange' if line.matched else 'red'
			draw.line( (utils.tup_int(line.pos), utils.tup_int( (line.pos[0] + line.length, line.pos[1]) )) ,
						width=int(line.thickness * 2), fill=color)
			draw.text( utils.tup_int(line.pos), "%.2f" % line.count, fill="black")
		for line in self.v_lines:
			color = 'purple' if line.matched else 'blue'
			draw.line( (utils.tup_int(line.pos), utils.tup_int( (line.pos[0], line.pos[1] + line.length) )) ,
						width=int(line.thickness * 2), fill=color)
			draw.text( utils.tup_int(line.pos), "%.2f" % line.count, fill="black")
		for line in self.text_lines:
			fill = colors[idx % len(colors)] if colortext else "black"
			draw.text(line.pos, line.text, font=utils.get_font(line.text, line.size[0]), fill=fill)
			draw.text( line.pos, "%.2f" % line.count, fill="blue")
			idx += 1

		return im
		
