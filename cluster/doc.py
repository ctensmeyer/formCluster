
import Image
import ImageDraw
import utils
import os

import components
import feature
from constants import *



def get_doc(_dir, source_file):
	document = Document(os.path.basename(source_file), os.path.join(_dir, source_file))
	return document

def get_docs(_dir, pr=True):
	'''
	returns (list(Document), int) - the list of Documents loaded and the
		number of incomplete documents.
		_dir - str the directory to load from.  Subdirectories are not considered
		pr - boolean to print or not
	'''
	docs = []
	num_loaded = 0
	num_exceptions = 0
	if pr:
		print "Loading Docs from:", _dir
	for f in os.listdir(_dir):
		if f.endswith(".txt"):
			try:
				docs.append(get_doc(_dir, f))
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
	'''
	returns (list(Document), int) - the list of Documents loaded and the
		number of incomplete documents.
		data_dir - str the top directory to load from.  All documents two 
			directories down are loaded.
		pr - boolean to print or not
	'''
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



class Document:
	'''
	Represents an element in our clustering scheme.  Composed of a single document
		digital image and all extracted features (contained in various files).
	Currently we use Text Lines, Horizontal Grid Lines, Vertical Grid Lines, and
		image dimensions as features
	'''

	def __init__(self, _id, f=None):
		'''
		_id - a unique id for the document
		f - str filename of the feature file.  Use None for self.copy()
		_original - boolean used by copy() to avoid reloading from files
		'''
		self._id = _id
		self.source_file = f
		self.loaded = False
		self.feature_sets = list()
		self.feature_set_names = list()
		self.feature_name_map = dict()
		if not LOAD_DOC_LAZY and self.source_file:
			self.load()
	
	def copy(self, new_id):
		''' Make a deep copy of a document with a new_id '''
		self._load_check()
		cpy = Document(new_id)
		cpy.loaded = True  # makes sure we never load data from files

		cpy.label = self.label
		cpy.size = self.size
		cpy.feature_set_names = self.feature_set_names[:]
		for feature_set in self.feature_sets:
			cpy_feature_set = feature_set.copy()
			cpy.feature_sets.append(cpy_feature_set)
			cpy.feature_name_map[cpy_feature_set.name()] = cpy_feature_set

		return cpy

	def _load_check(self):
		if not self.loaded:
			self.load()

	def load(self):
		f = open(self.source_file, 'r')

		# forgot to get the basename in the file in extraction script
		self._id = os.path.basename(f.readline().strip())

		self.label = f.readline().strip()

		size_line = f.readline().strip()
		tokens = size_line.split()
		self.size = ( int(tokens[0]), int(tokens[1]) )

		assert f.readline().strip() == ""

		if USE_TEXT:
			feature_set = feature.TextLineFeatureSet(self.size[0], self.size[1], REGION_ROWS, REGION_COLS, f)
			self.feature_sets.append(feature_set)
			name = feature_set.name()
			self.feature_set_names.append(name)
			self.feature_name_map[name] = feature_set
		else:
			utils.advance_to_blank(f)

		if USE_HORZ:
			feature_set = feature.GridLineFeatureSet(self.size[0], self.size[1], REGION_ROWS, REGION_COLS, components.Line.HORIZONTAL, f)
			self.feature_sets.append(feature_set)
			name = feature_set.name()
			self.feature_set_names.append(name)
			self.feature_name_map[name] = feature_set
		else:
			utils.advance_to_blank(f)

		if USE_VERT:
			feature_set = feature.GridLineFeatureSet(self.size[0], self.size[1], REGION_ROWS, REGION_COLS, components.Line.VERTICAL, f)
			self.feature_sets.append(feature_set)
			name = feature_set.name()
			self.feature_set_names.append(name)
			self.feature_name_map[name] = feature_set
		else:
			utils.advance_to_blank(f)

		if USE_SURF:
			feature_set = feature.SurfFeatureSet(self.size[0], self.size[1], REGION_ROWS, REGION_COLS, f)
			self.feature_sets.append(feature_set)
			name = feature_set.name()
			self.feature_set_names.append(name)
			self.feature_name_map[name] = feature_set
		else:
			utils.advance_to_blank(f)

		self.loaded = True

	def display(self):
		print "Doc: %s\tsize: %s" % (self._id, self.size)
		for feature_set in self.feature_sets:
			print
			print feature_set.name()
			feature_set.display()
		print

	def global_region_sim(self, other):
		''' returns all similarities as a vector '''
		self._load_check()
		other._load_check()
		return utils.flatten(self._feature_compare_helper(lambda fs1, fs2: fs1.global_region_sim(fs2), other, 'all'))

	def global_region_weights(self):
		''' returns default weights for all similarities '''
		#return utils.norm_list(utils.flatten(self._feature_compare_helper(lambda fs1, not_used: [1] + utils.flatten(fs1.region_weights()), self, 'all')))
		#return utils.norm_list(utils.flatten(map(lambda fs: [1] + utils.flatten(fs.region_weights()), self.feature_sets))) 
		self._load_check()
		weights = list()
		for fs in self.feature_sets:
			region_weights = fs.region_weights()
			weights.append([1] + utils.flatten(region_weights))
		return utils.norm_list(utils.flatten(weights))
			

	def global_sim(self, other, feature='all'):
		''' 
			returns the global sim score for the requested feature.
			feature='all' causes a vector of all global scores to be returned
		'''
		self._load_check()
		other._load_check()
		return self._feature_compare_helper(lambda fs1, fs2: fs1.global_sim(fs2), other, feature)

	def region_sim(self, other, feature='all'):
		''' 
			returns the region sim scores for the requested feature as a matrix.
			feature='all' causes a vector of all region scores to be returned
		'''
		self._load_check()
		other._load_check()
		return self._feature_compare_helper(lambda fs1, fs2: fs1.region_sim(fs2), other, feature)

	def region_weights(self, feature='all'):
		''' 
			returns the region weights for the requested feature as a matrix.
			feature='all' causes a vector of all region weights to be returned
		'''
		self._load_check()
		return self._feature_compare_helper(lambda fs1, not_used: fs1.region_weights(), self, feature)

	def region_sim_weights(self, other, feature='all'):
		''' 
			returns the region (sims, weights) for the requested feature as a matrix.
			feature='all' causes a vector of all region (sims, weights) to be returned
		'''
		self._load_check()
		other._load_check()
		return self._feature_compare_helper(lambda fs1, fs2: fs1.region_sim_with_weights(fs2), other, feature)

	def _feature_compare_helper(self, fun, other, feature):
		if feature == 'all':
			return map(fun, self.feature_sets, other.feature_sets)
		else:
			feature_set1 = self.get_feature_set(feature)
			feature_set2 = other.get_feature_set(feature)
			return fun(feature_set1, feature_set2)

	def get_feature_set(self, name):
		self._load_check()
		return self.feature_name_map[name]

	def get_feature_set_names(self):
		self._load_check()
		return self.feature_set_names

	def aggregate(self, other):
		'''
		Merge other into self
		Does not modify other, but self is modified
		'''
		self._load_check()
		other._load_check()

		self.size = (max(self.size[0], other.size[0]), max(self.size[1], other.size[1]))
		for feature_set1, feature_set2 in zip(self.feature_sets, other.feature_sets):
			feature_set1.aggregate(feature_set2)

		self.prune()

	def prune(self):
		'''
		Decrements each text/horz/vert line by the parameterized amount.  Remove lines
			that have counts below 0
		'''
		for feature_set in self.feature_sets:
			feature_set.prune()

	def final_prune(self):
		'''
		When clustering is done, remove text/horz/vert lines that have realatively very low
			counts
		'''
		for feature_set in self.feature_sets:
			feature_set.prune_final()


	def draw(self, colortext=False):
		'''
		:param colortext: True for drawing each text line a different color, False for all black
		:return Image:
		'''
		self._load_check()
		im = Image.new('RGB', self.size, 'white')
		draw = ImageDraw.Draw(im)
		for feature_set in self.feature_sets:
			feature_set.draw(draw)
		return im
		
