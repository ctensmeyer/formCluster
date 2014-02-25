
import Image
import ImageDraw
import math
import json
import lib_binarization as lb

# all functions should not modify the orginal image, but should return a copy

def profiles(im):
	im = greyscale(im)
	pix = im.load()
	horizontal = [0 for x in xrange(im.size[1])]
	vertical = [0 for x in xrange(im.size[0])]
	for x in xrange(im.size[0]):
		for y in xrange(im.size[1]):
			val = pix[x, y]
			vertical[x] += val
			horizontal[y] += val
	return {"Horizontal": horizontal, "Vertical": vertical}


def local_binary(im, threshold=127, ul=(0,0), size=None):
	'''
	Because this function is repeatedly called on an image, it does
		not return a copy, it modifies the original.
	'''
	if size is None:
		size = im.size
	pix = im.load()
	for x in xrange(ul[0], ul[0] + size[0]):
		for y in xrange(ul[1], ul[1] + size[1]):
			val = pix[x, y]
			new_val = 255 * int(val > threshold)
			pix[x, y] = new_val
	

def binary(im, threshold):
	return im.point(lambda x: 255 * int(x > threshold))

def global_otsu(im):
	'''
	returns the global binarization threshold according to Otsu's method
	'''
	im = greyscale(im)
	histo = lb.build_histogram({}, pixel_gen(im))
	distro = lb.normalize_histogram(histo)
	return lb.otsu(distro, 255)

	#print json.dumps(histo, indent=4)
	#return 127

all_thresholds = []
def local_otsu(im, hor_bins, ver_bins):
	'''
	:param im: Image
	:param hor_bins: number of bins across image
	:param ver_bins: number of bins up and down
	'''
	im = greyscale(im)
	width = im.size[0] / hor_bins  # need to deal with bin rounding
	height = im.size[1] / ver_bins

	for x in xrange(hor_bins):
		for y in xrange(ver_bins):
			_w = width 
			_h = height
			ul = (x * width, y * height)
			if x == (hor_bins - 1):
				# right margin, consume rest
				_w = im.size[0] - ul[0]
			if y == (ver_bins - 1):
				# bottom margin, consume rest
				_h = im.size[1] - ul[1]
			size = (_w, _h)
			local_histo = lb.build_histogram({}, pixel_gen(im, ul, size))
			local_distro = lb.normalize_histogram(local_histo)
			local_thresh = lb.otsu(local_distro, 255)
			#print local_thresh
			all_thresholds.append(local_thresh)
			local_binary(im, threshold=local_thresh, ul=ul, size=size)
	return im


def display_all_thresh():
	print json.dumps(lb.build_histogram({}, all_thresholds), indent=4)
	

def greyscale(im):
	return im.convert("L")


def pixel_gen(im, ul=(0,0), size=None):
	'''
	:param im: Image
	:param ul: int-pair coordinates of the upper left corner of box
	:param size: int-pair (width, length) of window to iterate over
	'''
	if size is None:
		size = im.size
	pix = im.load()
	for x in xrange(ul[0], ul[0] + size[0]):
		for y in xrange(ul[1], ul[1] + size[1]):
			yield pix[x, y]


def median_filter(im, ker_width, ker_height):
	'''
	:param ker_width: width of median kernal
	:param ker_height: height of median kernal
	'''
	#TODO: this is the naive implementation. More efficient
	im = im.convert('L')
	pix_orig = im.load()
	im_new = im.copy()
	pix_new = im_new.load()
	im_width = im.size[0]
	im_height = im.size[1]
	for x in xrange(im_width):
		for y in xrange(im_height):
			vals = []
			for _x in xrange(-1 * ker_width / 2, (ker_width + 1) / 2):
				for _y in xrange(-1 * ker_height / 2, (ker_height + 1) / 2):
					x_i = x + _x
					y_i = y + _y
					if in_range(x_i, 0, im_width-1) and in_range(y_i, 0, im_height-1):
						vals.append(pix_orig[x_i, y_i])
			vals.sort()
			idx = len(vals) / 2
			new_val = vals[idx]
			pix_new[x, y] = new_val
	return im_new
			

# caller is responsible that im has the right dimensions
def graph_profile(im, profile, color='black', width=1):
	'''
	Graph a sequence on an existing image
	:param im: Image to draw on.  Original unmodified
	:param profile: list.  Elements are rounded to nearest int
	:param color: color of the line to draw
	:param width: width of the line to draw
	:return: copy of im with the line drawn on it
	'''
	im = im.copy()
	height = im.size[1]
	draw = ImageDraw.Draw(im)
	points = [(x, height - int(round(y))) for x, y in enumerate(profile)]
	draw.line(points, fill=color, width=width)
	
	return im
	

def graph_points(im, points, color='black', width=1):
	'''
	Graph some on an existing image
	:param im: Image to draw on.  Original unmodified
	:param profile: list.  Elements are rounded to nearest int
	:param color: color of the points to draw
	:param width: diameter of the points to draw
	:return: copy of im with the points drawn on it
	'''
	im = im.copy()
	height = im.size[1]
	draw = ImageDraw.Draw(im)
	circ = width / 2
	for point in points:
		x = int(round(point[0]))
		y = int(round(point[1]))
		bounding = [x-circ, height-(y+circ), x+circ, height-(y-circ)]
		draw.ellipse(bounding, fill=color)

	return im


def graph_matching(im, matching, ex_p1, ex_p2, color='black', width=1):
	'''
	Graph lines between points.
	:param im: Image to draw on.  Original unmodified
	:param matching: A sequence of triples (idx1, idx2, label)
	:ex_p1: A sequence of (x, y)
	:ex_p2: A sequence of (x, y)
	:color: Color of lines
	:width: Width of line in pixels
	:return: copy of im with lines drawn between points in ex_p1 and points in ex_p2 according to matching
	'''
	im = im.copy()
	im_height = im.size[1]
	draw = ImageDraw.Draw(im)
	for match in matching:
		idx1 = match[0]
		idx2 = match[1]
		p1 = ex_p1[idx1] 
		p2 = ex_p2[idx2] 

		# point transform
		p1 = (p1[0], im_height - p1[1])
		p2 = (p2[0], im_height - p2[1])

		draw.line([p1, p2], fill=color, width=width)

		# draw label above p2
		label_p = (p2[0], p2[1] - 50)
		label = str(match[2])
		if len(label) > 6:
			label = label[:6]
		draw.text(label_p, label, fill='black')
	return im
		

_2rootpi = math.sqrt(2 * math.pi)
def gauss(x, sig, mu=0):
	const = 1.0 / (sig * (_2rootpi))
	return const * math.exp( -( (x - mu) ** 2) / (2. * sig * sig ))
	

def _init_gauss_dist_table(win_size, space_sig):
	table = []
	for x in xrange(win_size / 2 + 1):
		row = []
		for y in xrange(win_size / 2 + 1):
			dist = math.sqrt(x ** 2 + y ** 2)
			val = gauss(dist, space_sig)
			row.append(val)
		table.append(row)
	return table
	

# This is really slow for python.  It uses brute force.  O(image_size * win_size^2)
def bilateral(im, win_size, value_sig, space_sig): 
	im = im.copy()
	width, height = im.size
	out = Image.new("L", im.size)
	pix_in = im.load()
	pix_out = out.load()
	gauss_dist_table = _init_gauss_dist_table(win_size, space_sig)
	for x in xrange(width):
		if x % (width / 10) == 0:
			print x
		for y in xrange(height):
			pix_old = pix_in[x, y]
			pix_sum = 0.0
			weight_sum = 0.0
			for x_off in xrange(-1 * (win_size / 2), (win_size / 2) + 1):
				if not in_range(x + x_off, 0, width-1):
					continue
				for y_off in xrange(-1 * (win_size / 2), (win_size / 2) + 1):
					if not in_range(y + y_off, 0, height-1):
						continue
					pix_new = pix_in[x + x_off, y + y_off]
					# optimize via lookup table
					#dist = math.sqrt(x_off ** 2 + y_off ** 2)
					diff = abs(pix_old - pix_new)
					gauss_dist = gauss_dist_table[abs(x_off)][abs(y_off)] # gauss(dist, space_sig)
					gauss_diff = gauss(diff, value_sig)
					weight = gauss_dist * gauss_diff
					pix_sum += weight * pix_new
					weight_sum += weight
			pix_out[x, y] = (1 / weight_sum) * pix_sum
	return out
	

def in_range(val, _min, _max):
	return val >= _min and val <= _max


if __name__ == "__main__":
	pass
	


