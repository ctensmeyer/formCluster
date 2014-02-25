
import xml.etree.ElementTree as ET
import os
import ast
import sys
import json
import math
import signalutils
import imageutils
import Image


_dictionary = set()
_dict_path = '/usr/share/dict/american-english'
def load_dict():
	for line in open(_dict_path).readlines():
		_dictionary.add(line.strip())
	
load_dict()


def pos_to_enum(p, dim):
	e = ''
	if p[1] < 0.33 * dim[1]:
		e += 'U'
	elif p[1] < 0.67 * dim[1]:
		e += 'M'
	else:
		e += 'L'

	if p[0] < 0.33 * dim[0]:
		e += 'L'
	elif p[0] < 0.67 * dim[0]:
		e += 'M'
	else:
		e += 'R'
	
	return e

def load_bounding_boxes(infilepath):
	tree = ET.parse(infilepath)
	bounding_boxes = {}
	for char in tree.iter(tag='*'):
		if char.tag.endswith('charParams'):
			if not char.get('suspicious'):
				l = int(char.get('l'))
				t = int(char.get('t'))
				r = int(char.get('r'))
				b = int(char.get('b'))
				bounding_boxes[(l, t, r, b)] = char.text
	return bounding_boxes

def _extract_characters(node, include_suspicious=True):
	s = ''
	total_line_height = 0
	for char in node.iter(tag='*'):
		if char.tag.endswith('charParams'):
			if include_suspicious or not char.get('suspicious'):
				s += char.text
				t = int(char.get('t'))
				b = int(char.get('b'))
				line_height = b - t
				total_line_height += line_height
	average_line_height = total_line_height / float(len(s)) if s else 0
	return s, round(average_line_height, 3)

def _get_center(line):
	l = int(line.get('l'))
	t = int(line.get('t'))
	r = int(line.get('r'))
	b = int(line.get('b'))
	return  (l + r) / 2 , (t + b) / 2


def filter_titles(possible_titles):
	new_arr = []
	# generous filtering...
	for title in possible_titles:
		line = title[1]
		if len(line) > 5 and any(map(lambda word: word.lower() in _dictionary and len(word) > 3, line.split())):
			new_arr.append(title)
	return new_arr
	

def find_title(infilepath):
	'''
	Currently based on size
	'''
	d = {'title': '', 'title_pos': None, 'subtitle': '', 'subtitle_pos': None}
	possible_titles = []
	tree = ET.parse(infilepath)
	for node in tree.iter(tag='*'):
		if node.tag.endswith('line'):
			line, avg_line_height = _extract_characters(node)
			possible_titles.append( (avg_line_height, line, node) )
	possible_titles = filter_titles(possible_titles)
	possible_titles.sort(reverse=True)
	for title in possible_titles:
		#print title[:-1]
		pass
	
	# check if the largest is significantly larger
	if len(possible_titles) > 3:
		first = possible_titles[0][0]
		second = possible_titles[1][0]
		if first - second > 3:
			d['title'] = possible_titles[0][1]
			d['title_pos'] = pos_to_enum(_get_center(possible_titles[0][2]), (6384, 3744))
	return d


def load_all_profiles(indir, preprocess={}, normalize=True):
	'''
	:param indir: str dir_path for .xml files containing profile information
	:param preprocess: func (list -> list) for preprocessing profiles 
	:return: dict {<filename> : {'Horizontal' : <profile>, 'Vertical' : <profile>, 'dimensions' : (width, height)}, }
	'''
	all_profiles = {}
	for filename in os.listdir(indir):
		if not filename.endswith(".xml"):
			continue
		filepath = os.path.join(indir, filename)
		if os.path.isfile(filepath):
			all_profiles[filename] = get_profiles(filepath, preprocess, normalize)
	return all_profiles
	

def calc_all_profiles(indir, preprocess={}, normalize=True):
	'''
	:param indir: str dir_path for .jpg images
	:param preprocess: func (list -> list) for preprocessing profiles 
	:return: dict {<filename> : {'Horizontal' : <profile>, 'Vertical' : <profile>, 'dimensions' : (width, height)}, }
	'''
	all_profiles = {}
	for filename in os.listdir(indir):
		if not filename.endswith(".jpg"):
			continue
		filepath = os.path.join(indir, filename)
		if os.path.isfile(filepath):
			all_profiles[filename] = calc_profiles(filepath, preprocess, normalize)
	return all_profiles


def calc_profiles(infilepath, preprocess={}, normalize=True):
	im = Image.open(infilepath)
	profiles = imageutils.profiles(im)
	profiles['dimensions'] = im.size
	for x, _type in enumerate(['Horizontal', 'Vertical']):
		if normalize:
			profiles[_type] = signalutils.scale(profiles[_type], 1.0 / profiles['dimensions'][x])
		if preprocess:
			func = preprocess.get(_type)
			if not func:
				func = preprocess.get('all')
			if func:
				profiles[_type] = func(profiles[_type])

	return profiles


def get_profiles(infilepath, preprocess={}, normalize=True):
	'''
	:param infilepath: str file_path for .xml file containing profile information
	:param preprocess: func (list -> list) for preprocessing profiles
	:return: dict {'Horizontal' : <profile>, 'Vertical' : <profile>, 'dimensions' : (width, height)}
	'''
	root = ET.parse(infilepath).getroot()
	profiles = {}
	dimensions = (0, 0)
	for ele in list(root):
		if ele.tag == "Profile":
			values = ast.literal_eval(ele.get('values'))
			_type = ele.get('type')
			profiles[_type] = values
		if ele.tag == "ImageInfo":
			dimensions = ( int(ele.get('width')), int(ele.get('height')) )
			profiles['dimensions'] = dimensions
	if normalize:
		profiles['Horizontal'] = signalutils.scale(profiles['Horizontal'], 1.0 / profiles['dimensions'][0])
		profiles['Vertical'] = signalutils.scale(profiles['Vertical'], 1.0 / profiles['dimensions'][1])
	if preprocess:
		for _type in ['Horizontal', 'Vertical']:
			func = preprocess.get(_type)
			if not func:
				func = preprocess.get('all')
			if func:
				profiles[_type] = func(profiles[_type])
	return profiles


tmp_dir = "tmp/"
def form_sim_mat(all_profiles):
	'''
	:param all_profiles: dict: {identifier : {'Horizontal' : <profile>, 'Vertical' : <profile>, 'dimensions' : (width, height)}, }
	:return: list of lists of similarity scores (float).  Row/Col ordering is sorted order of identifiers
	'''
	sim_mat = []
	for id_1 in sorted(all_profiles.keys()):
		row = []
		for id_2 in sorted(all_profiles.keys()):
			if id_1 == id_2:
				row.append(0.0)
				continue
			print id_1, id_2
			_sum = 0
			for _type in ['Vertical']:
				pr_1 = all_profiles[id_1][_type]
				pr_2 = all_profiles[id_2][_type]
				ex_1 = signalutils.extract_extrema(pr_1)
				ex_2 = signalutils.extract_extrema(pr_2)

				#min_support_1 = len(pr_1) / 100
				#deltas_1 = map(lambda x: abs(x), signalutils.extract_delta(pr_1, ex_1))
				#min_delta_1 = signalutils.avg(deltas_1) - 0.5 * signalutils.stdev(deltas_1)

				min_support_1 = 0
				min_delta_1 = 3
				f_ex_1 = signalutils.filter_extrema(pr_1, ex_1, min_support_1, min_delta_1)

				#min_support_2 = len(pr_2) / 100
				#deltas_2 = map(lambda x: abs(x), signalutils.extract_delta(pr_2, ex_2))
				#min_delta_2 = signalutils.avg(deltas_2) - 0.5 * signalutils.stdev(deltas_2)

				min_support_2 = 0
				min_delta_2 = 3
				f_ex_2 = signalutils.filter_extrema(pr_2, ex_2, min_support_2, min_delta_2)

				#write_data(os.path.join(tmp_dir, 'pr1.plot'), pr_1)
				#write_data(os.path.join(tmp_dir, 'pr2.plot'), pr_2)

				#write_data(os.path.join(tmp_dir, 'ex1.plot'), signalutils.extract_delta(pr_1, ex_1), ex_1)
				#write_data(os.path.join(tmp_dir, 'ex2.plot'), signalutils.extract_delta(pr_2, ex_2), ex_2)

				#write_data(os.path.join(tmp_dir, 'f_ex1.plot'), signalutils.extract_delta(pr_1, f_ex_1), f_ex_1)
				#write_data(os.path.join(tmp_dir, 'f_ex2.plot'), signalutils.extract_delta(pr_2, f_ex_2), f_ex_2)

				_sum += signalutils.extrema_distance(pr_1, f_ex_1, pr_2, f_ex_2)[0] ** 2
			row.append(math.sqrt(_sum))
		sim_mat.append(row)
	return sim_mat


def write_cols(outfile, data):
	'''
	Write out data in row/col format
	:param outfile: str filename
	:param data: list of tuples. Each tuple is a row.
	'''
	with open(outfile, 'w') as out:
		for x in xrange(len(data)):
			# this is why I love python
			out.write("%s\n" % " ".join(map(str, data[x])))
		out.flush()
		out.close()


def write_data(outfile, data, labels=None):
	'''
	:param outfile: str filename
	:param data: list of data to write 
	:param labels: list of optional labels
	'''
	with open(outfile, 'w') as out:
		for x in xrange(len(data)):
			if labels:
				out.write("%s %s %s\n" % (x, labels[x], data[x]))
			else:
				out.write("%s %s\n" % (x, data[x]))
		out.flush()
		out.close()
			

def _pre_hor(profile):
	blur_amount = len(profile) / 100
	profile = signalutils.blur_uniform(profile, blur_amount)
	profile = signalutils.blur_uniform(profile, blur_amount)
	return profile


def _pre_ver(profile):
	blur_amount = len(profile) / 200
	profile = signalutils.blur_uniform(profile, blur_amount)
	return profile
	
def bilateral(prof):
	window = len(prof) / 200
	value_sigma = 20
	spacial_sigma = len(prof) / 200.0
	prof = signalutils.blur_bilateral(prof, window, spacial_sigma, value_sigma)
	prof = signalutils.blur_bilateral(prof, window, spacial_sigma, value_sigma)
	return prof

def _preprocess(profile):
	'''
	Preprocessing step to double blur the profiles
	'''
	window = len(profile) / 100
	value_sigma = 8
	spacial_sigma = len(profile) / 250.0
	#blur_amount = len(profile) / 100
	#profile = signalutils.blur_uniform(profile, blur_amount)
	profile = signalutils.blur_bilateral(profile, window, spacial_sigma, value_sigma)
	#return signalutils.blur_uniform(profile, blur_amount)
	return profile

if __name__ == "__main__":
	#if len(sys.argv) < 3:
	#	raise Exception("[image_dir sim_mat.txt]")

	#processors = {'Horizontal': _pre_hor, 'Vertical': _pre_ver}
	#processors = {'all': bilateral}
	#profiles = calc_all_profiles(sys.argv[1], processors, normalize=True)
	#sim_mat = form_sim_mat(profiles)
	#write_data(sys.argv[2], sim_mat, sorted(profiles.keys()))
	_dir = sys.argv[1]
	for _f in sorted(os.listdir(_dir)):
		if not _f.endswith('.xml'):
			continue
		infilepath = os.path.join(_dir, _f)
		print _f, find_title(infilepath)
	
