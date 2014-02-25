
import ocr
import Image
import json
import profiles
import string
import utils
from components import TextLine

LAZY = True
ALLOW_PARTIAL_MATCHES = False
DEBUG = False

class Document:
	
	def __init__(self, _id, image_path, ocr_path, prof_path, form_path, original=True):
		self._id = _id
		self.image_path = image_path
		self.ocr_path = ocr_path
		self.prof_path = prof_path # path of projection profiles
		self.image_path = image_path
		self.form_path = form_path

		self.loaded = False
		if not LAZY and original:
			self.load()

	def copy(self, new_id):
		self._load_check()
		cpy = Document(new_id, self.image_path, self.ocr_path, self.prof_path, self.form_path, original=False)
		cpy.loaded = True  # makes sure we never load data from files

		cpy.label = self.label
		cpy.text_lines = list()
		for line in self.text_lines:
			cpy.text_lines.append(line.copy())
		cpy.size = self.size

		cpy.char_mass = self.char_mass

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
				if ALLOW_PARTIAL_MATCHES and match != TextLine.COMPLETE_MATCH:
					if match == TextLine.PREFIX_MATCH:
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
                    
                    
	def text_line_distance(self, other):
		self._load_check()
		other._load_check()
		thresh_dist = 0.10 * max(max(self.size), max(other.size))  # % of largest dimension

		self.clear_matches()
		other.clear_matches()

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

	def clear_matches(self):
		for line in self.text_lines:
			line.matched = False

	def aggregate(self, other):
		self._load_check()
		other._load_check()
		thresh_dist = 0.10 * max(max(self.size), max(other.size))  # % of largest dimension
		
		self.clear_matches()
		other.clear_matches()

		to_add = list()
		for line in other.text_lines:
			matched_line = self._find_matching_text_line(line, thresh_dist)
			if matched_line:
				x = (line.count * line.pos[0] + matched_line.count * matched_line.pos[0]
					) / (line.count + matched_line.count)
				y = (line.count * line.pos[1] + matched_line.count * matched_line.pos[1]
					) / (line.count + matched_line.count)

				# Don't worry about size for now
				#width = (line.count * line.size[0] + matched_line.count * matched_line.size[0]) /
				#	(line.count + matched_line.count)
				#height = (line.count * line.size[1] + matched_line.count * matched_line.size[1]) /
				#	(line.count + matched_line.count)
				# hmmm, what if the text is not an exact match...
				# partial matches?

				# translations
				dx = x - matched_line.pos[0]
				dy = y - matched_line.pos[1]
				for char in matched_line.chars:
					new_pos = (char.pos[0] + dx, char.pos[1] + dy)
					char.pos = new_pos
					char.pos2 = (char.pos2[0] + dx, char.pos2[1] + dy)
					#char.attributes['l'] += dx
					#char.attributes['r'] += dx
					#char.attributes['t'] += dy
					#char.attributes['b'] += dy

				matched_line.count += line.count
				matched_line.pos = (x, y)
				#matched_line.size = (width, height)
			else:
				to_add.append(line)

		self.text_lines += to_add
		self._set_total_char_mass()
			
