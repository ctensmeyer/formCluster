
import Image
from collections import deque
import sys


# Enum
START = 0
BETWEEN = 1
CONSEC = 2
GAP = 3
LABELING = 4
END = 5


# Constants
BLACK = 0
WHITE = 255


# Configurations
CONSEC_THRESHOLD = 70
MAX_GAP = 3
COLOR_LABEL = (0, 255, 0)
SMOOTH_KERNEL = 7


def label_component(x, y, pix, mask, label):
	pix_queue = deque([(x, y)])
	pix[x, y] = label
	cc_coords = []

	while pix_queue:
		x, y = pix_queue.popleft()
		for _x in [-1, 0, 1]:
			for _y in [-1, 0, 1]:
				__x = x + _x
				__y = y + _y
				try:
					if mask[__x, __y] not in [BLACK, WHITE]:
						print __x, __y, mask[__x, __y]
					if pix[__x , __y] != label and mask[__x, __y] == BLACK:
						pix[__x, __y] = label
						cc_coords.append( (__x, __y) )
						pix_queue.append( (__x, __y) )
				except:
					pass


def find_ccs(im):
	pix = im.load()
	mut = Image.new('I', im.size, color='black') # assign all 0s
	mut_pix = mut.load()
	cur_idx = 1
	for x in xrange(im.size[0]):
		for y in xrange(im.size[1]):
			if pix[x, y] == BLACK and mut_pix[x, y] == BLACK:
				label_component(x, y, mut_pix, pix, cur_idx)
				cur_idx += 1
	print cur_idx
	return mut
	

def color_components(orig, ccs, colors):
	n = len(colors)
	#mut = Image.new('RGB', o.size, color='white') 
	mut = orig.convert('RGB')
	pix = ccs.load()
	mut_pix = mut.load()
	for x in xrange(orig.size[0]):
		for y in xrange(orig.size[1]):
			if pix[x, y] != 0:
				mut_pix[x, y] = colors[ pix[x, y] % n]

	return mut

def get_value(pix, x, y):
	try:
		return pix[x, y]
	except:
		return 0


def neighbors(pix, x, y):
	vals = []
	for _x in [-1, 0, 1]:
		for _y in [-1, 0, 1]:
			vals.append(get_value(pix, x + _x, y + _y))
	return vals


def nebs(x, y, x_size, y_size, pix, mask):
	vals = []
	for _x in xrange(-1 * x_size / 2, (x_size + 1) / 2):
		for _y in xrange(-1 * y_size / 2, (y_size + 1) / 2):
			__x = x + _x
			__y = y + _y
			try:
				if mask[__x, __y] == BLACK:
					vals.append(pix[__x, __y])
			except:
				pass
	return vals
		

def smooth_line_data(im, horz_lines):
	'''
	Propagates line data to adjacent pixels
	'''
	copy = horz_lines.copy()
	pix = im.load()
	line_pix = horz_lines.load()
	copy_pix = copy.load()
	for x in xrange(im.size[0]):
		for y in xrange(im.size[1]):
			if pix[x, y] == BLACK and copy_pix[x, y] <= CONSEC_THRESHOLD:
				if max(neighbors(copy_pix, x, y)) > CONSEC_THRESHOLD:
					line_pix[x, y] = CONSEC_THRESHOLD + 1


def median(arr):
	'''
	returns median of the list
	'''
	sorts = sorted(arr)
	n = len(sorts)
	if n % 2 == 0:
		return (sorts[n / 2] + sorts[n / 2 - 1]) / 2.0
	return sorts[n / 2]


def smooth_median(im, lines):
	'''
	smooths the line data by performing a median filter over it
		using im as a mask
	'''
	copy = lines.copy()
	pix = im.load()
	line_pix = lines.load()
	copy_pix = copy.load()

	for x in xrange(im.size[0]):
		for y in xrange(im.size[1]):
			if pix[x, y] == BLACK and copy_pix[x, y] <= CONSEC_THRESHOLD:
				neb_vals = nebs(x, y, 20, 7, copy_pix, pix)
				new_val = median(neb_vals)
				line_pix[x, y] = new_val
			
			
def label_horz_lines(im):
	'''
	Takes an Image (thresholded) and returns an 'I' image with the horizontal
	consecutive regions labeled
	'''
	im = im.convert('1')
	width, height = im.size
	mut = Image.new('I', im.size, color='black') # assign all 0s
	pix_orig = im.load()
	pix_mut = mut.load()

	x = 0 # iterator index
	_x = 0 # marks start of consecutive region
	y = 0 # iterator index
	gapped = 0
	consec = 0

	for y in xrange(height):
		state = START
		x = 0
		while state != END:
			#print "(%d, %d)" % (x, y)

			if state == START:
				# decide which state to go to
				if pix_orig[x, y] == BLACK:
					consec = 0
					state = CONSEC
					_x = x
				elif pix_orig[x, y] == WHITE:
					state = BETWEEN
				else:
					print "pix_orig[%d, %d] = %s" % (x, y, pix_orig[x, y])
					assert(False)

			elif state == BETWEEN:
				# burn through pixels
				if pix_orig[x, y] == BLACK:
					state = CONSEC
					consec = 0
					_x = x # record the start of a new consecutive block
				elif pix_orig[x, y] == WHITE:
					x += 1
					if x >= width:
						state = END
				else:
					print "pix_orig[%d, %d] = %s" % (x, y, pix_orig[x, y])
					assert(False)

			elif state == CONSEC:
				# burn through pixels, detect gap and end
				if pix_orig[x, y] == BLACK:
					consec += 1
					x += 1
					if x >= width:
						state = LABELING
				elif pix_orig[x, y] == WHITE:
					gapped = 0
					state = GAP
				else:
					print "pix_orig[%d, %d] = %s" % (x, y, pix_orig[x, y])
					assert(False)

			elif state == GAP:
				# go for a few pixels to see if we get BLACK again
				if pix_orig[x, y] == BLACK:
					state = CONSEC
				elif pix_orig[x, y] == WHITE:
					x += 1
					gapped += 1
					if x >= width or gapped > MAX_GAP:
						state = LABELING
				else:
					print "pix_orig[%d, %d] = %s" % (x, y, pix_orig[x, y])
					assert(False)

			elif state == LABELING:
				for __x in xrange(_x, x):
					if pix_orig[__x, y] == BLACK:
						pix_mut[__x, y] = consec
				if x >= width:
					state = END
				else:
					state = START
			elif state == END:
				continue # exit condition
			else:
				print "State is %d" % state
				assert(False)
	# end loops
	return mut


def paint_lines(canvas, lines):
	'''
	:param canvas: Image 'RGB' to paint on.  This is modified
	:param lines: Image 'I' line data.  This is not modified
	'''
	pix_canvas = canvas.load()
	pix_lines = lines.load()
	for x in xrange(lines.size[0]):
		for y in xrange(lines.size[1]):
			if pix_lines[x, y] > CONSEC_THRESHOLD:
				pix_canvas[x, y] = COLOR_LABEL


def line_detect(im, thresh=None, noise_gap=None, color=None):
	'''
	Performs rudimentary line detection
	:param im: binary Image(L), not modified
	:param thresh: int minimum line length
	:param noise_gap: int minimum number of pixels between separate contiguous regions
	:param color: color to mark the lines in the returned Image
	'''
	if thresh is not None:
		CONSEC_THRESHOLD = thresh
	if noise_gap is not None:
		MAX_GAP = noise_gap
	if color is not None:
		COLOR_LABEL = color

	im = im.copy()
	rotated = im.rotate(90)
	horz_lines = label_horz_lines(im)
	vert_lines = label_horz_lines(rotated)
	print "raw lines done"
	#for x in xrange(1):
	#	smooth_median(rotated, vert_lines)
	#	smooth_median(im, horz_lines)
	#	print "done with round of median filter"
	vert_lines = vert_lines.rotate(270)
	canvas = im.convert('RGB')
	paint_lines(canvas, horz_lines)
	paint_lines(canvas, vert_lines)
	return canvas


