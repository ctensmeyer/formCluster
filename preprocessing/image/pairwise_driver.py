
import Image
import sys
import os
import imageutils as imut
import signalutils as sigut

_types = ['Horizontal', 'Vertical']

def scale_profiles(p, size):
	width = size[0]
	height = size[1]
	for x,_type in enumerate(_types):
		p[_type] = sigut.scale(p[_type], 1.0 / size[x])


def preprocess_profiles(prof, f):
	for _type in _types:
		prof[_type] = f(prof[_type])
	

def bilateral(prof):
	window = len(prof) / 200
	value_sigma = 20
	spacial_sigma = len(prof) / 200.0
	prof = sigut.blur_bilateral(prof, window, spacial_sigma, value_sigma)
	prof = sigut.blur_bilateral(prof, window, spacial_sigma, value_sigma)
	return prof


def uniform(prof):
	window = len(prof) / 100
	prof = sigut.blur_uniform(prof, window)
	return prof


def filter_extrema(ex, p):
	 return sigut.filter_extrema(p, ex, 0, 8)
	

def main(image_dir, out_dir, idx1, idx2):
	# get images
	f1 = "im%d.jpg" % idx1
	f2 = "im%d.jpg" % idx2
	fp1 = os.path.join(image_dir, f1)
	fp2 = os.path.join(image_dir, f2)
	im1 = Image.open(fp1)
	im2 = Image.open(fp2)
	im1 = im1.convert('RGB')
	im2 = im2.convert('RGB')

	# calculate profiles
	p1 = imut.profiles(im1)
	p2 = imut.profiles(im2)

	# normalize
	scale_profiles(p1, im1.size)
	scale_profiles(p2, im2.size)

	# perform signal preprocessing bilateral, etc
	preprocess_profiles(p1, bilateral)
	preprocess_profiles(p2, bilateral)

	for x,_type in enumerate(_types):
		# determine extrema
		# filter extrema
		# graph on both images...
		# im1 prof will be blue.  im2 prof will be green
		# Filtered out extrema will be orange
		# Kept extrema will be black
		# Lines between matching extrema will be yellow
		_p1 = sigut.scale(p1[_type], 10)
		_p2 = sigut.scale(p2[_type], 10)
		_im1 = im1.copy()
		_im2 = im2.copy()

		if _type == 'Horizontal':
			# rotate images for drawing
			_im1 = _im1.rotate(90)
			_im2 = _im2.rotate(90)

		# extrema indices
		ex1 = sigut.extract_extrema(_p1)
		ex2 = sigut.extract_extrema(_p2)

		# extrema points 
		pt1 = sigut.extrema_as_points(ex1, _p1)
		pt2 = sigut.extrema_as_points(ex2, _p2)

		# filtered extrema indices
		fex1 = filter_extrema(ex1, _p1)
		fex2 = filter_extrema(ex2, _p2)

		# filtered extrema points 
		fpt1 = sigut.extrema_as_points(fex1, _p1)
		fpt2 = sigut.extrema_as_points(fex2, _p2)

		# matching extrema indices and cost
		cost, matching = sigut.extrema_distance(_p1, fex1, _p2, fex2)

		# draw profile 1 & 2
		_im1 = imut.graph_profile(_im1, _p1, color='blue', width=5)
		_im1 = imut.graph_profile(_im1, _p2, color='green', width=5)

		# draw matching lines
		_im1 = imut.graph_matching(_im1, matching, fpt1, fpt2, color='purple', width=3)

		# draw unfiltered extrema
		_im1 = imut.graph_points(_im1, pt1, color='orange', width=7)
		_im1 = imut.graph_points(_im1, pt2, color='orange', width=7)
		
		# draw filtered extrema
		_im1 = imut.graph_points(_im1, fpt1, color='black', width=7)
		_im1 = imut.graph_points(_im1, fpt2, color='black', width=7)

		# draw profile 1 & 2
		_im2 = imut.graph_profile(_im2, _p1, color='blue', width=5)
		_im2 = imut.graph_profile(_im2, _p2, color='green', width=5)

		# draw matching lines
		_im2 = imut.graph_matching(_im2, matching, fpt1, fpt2, color='purple', width=3)

		# draw unfiltered extrema
		_im2 = imut.graph_points(_im2, pt1, color='orange', width=7)
		_im2 = imut.graph_points(_im2, pt2, color='orange', width=7)
		
		# draw filtered extrema
		_im2 = imut.graph_points(_im2, fpt1, color='black', width=7)
		_im2 = imut.graph_points(_im2, fpt2, color='black', width=7)

		if _type == 'Horizontal':
			# rotate images back
			_im1 = _im1.rotate(270)
			_im2 = _im2.rotate(270)

		# file system stuff
		dir_name = "%d_%d" % (idx1, idx2)
		dir_path = os.path.join(out_dir, dir_name)
		try:
			os.makedirs(dir_path)
		except:
			pass
		# filenames
		fn1 = "%d_%s.jpg" % (idx1, _type)
		fn2 = "%d_%s.jpg" % (idx2, _type)

		# filepaths
		fp1 = os.path.join(dir_path, fn1) 
		fp2 = os.path.join(dir_path, fn2) 

		# save images
		_im1.save(fp1)
		_im2.save(fp2)


if __name__ == "__main__":
	print sys.argv
	if len(sys.argv) < 5:
		raise Exception("[image_dir out_dir idx1 idx2]")
	image_dir = sys.argv[1]
	out_dir = sys.argv[2]
	idx1 = int(sys.argv[3])
	idx2 = int(sys.argv[4])
	main(image_dir, out_dir, idx1, idx2)

