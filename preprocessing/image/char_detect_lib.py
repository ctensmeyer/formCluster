
import Image
import ImageDraw
import sys

BLACK = 0
WHITE = 1


def in_range(p, bound):
	return p[0] < bound[0] and p[1] < bound[1]

def top_most(im, color=(255, 0, 0)):
	pix = im.load()
	for y in xrange(im.size[1]):
		for x in xrange(im.size[0]):
			if pix[x, y] == color:
				return y
	return im.size[1]

def left_most(im, color=(255, 0, 0)):
	#return top_most(im.rotate(90), color)
	pix = im.load()
	for x in xrange(im.size[0]):
		for y in xrange(im.size[1]):
			if pix[x, y] == color:
				return x
	return im.size[0]

def crop_special(im):
	l = left_most(im)
	u = top_most(im)
	r = im.size[0]
	b = im.size[1]

	return im.crop((l, u, r, b))


def xor_images(im1, im2,
					overlap_color=(0, 0, 255),
					im1_color=(255, 0, 0),
					im2_color=(0, 255, 0)):
	size = (max(im1.size[0], im2.size[0]), max(im1.size[1], im2.size[1]))
	mut = Image.new('RGB', size, 'white')
	pix_mut = mut.load()
	pix1 = im1.load()
	pix2 = im2.load()
	width = size[0]
	height = size[1]
	for x in xrange(width):
		for y in xrange(height):
			if in_range( (x, y), im1.size) and in_range( (x, y), im2.size):
				if pix1[x, y] == pix2[x, y]:
					if pix1[x, y] != (255, 255, 255):  # white
						pix_mut[x, y] = overlap_color
					else:
						pass # mut is white by default
				else:
					if pix1[x, y] == (255, 255, 255):  # white
						pix_mut[x, y] = im2_color
					elif pix2[x, y] == (255, 255, 255):  # white
						pix_mut[x, y] = im1_color
					else:
						pix_mut[x, y] = (0, 0, 0)  # black
			elif in_range( (x, y), im1.size):
				pix_mut[x, y] = im1_color
			elif in_range( (x, y), im2.size):
				pix_mut[x, y] = im2_color
			else:
				print "we have a problem at %s" % ((x, y),)
	return mut


def char_detect(im, bounding_boxes, color=(255, 0, 0)):
	'''
	im is L
	'''
	mut = im.convert('RGB')
	pix = im.load()
	mut_pix = mut.load()
	draw = ImageDraw.Draw(mut)
	for box in bounding_boxes:
		#draw.rectangle(box, outline=color)
		
		l, t, r, b = box	
		for x in xrange(l, r):
			for y in xrange(t, b):
				if pix[x, y] == BLACK:
					mut_pix[x, y] = color
	return mut

def merge_with_line(im_char, im_line,
						  overlap_color=(127, 127, 0),
						  char_color=(255, 0, 0),
						  line_color=(0, 255, 0)):
	'''
	Assumes input images are RBG
	'''
	im_char = im_char.copy()  # don't modify the original
	pix_char = im_char.load()
	pix_line = im_line.load()
	width = im_char.size[0]
	height = im_char.size[1]

	for x in xrange(width):
		for y in xrange(height):
			if pix_line[x, y] == line_color:
				pix_char[x, y] = overlap_color if pix_char[x, y] == char_color else line_color

	return im_char
	

def compare(im1, im2):
	im1 = crop_special(im1)
	im2 = crop_special(im2)
	return xor_images(im1, im2)

if __name__ == "__main__":
	im1 = crop_special(Image.open(sys.argv[1]))
	im2 = crop_special(Image.open(sys.argv[2]))
	im1.save("tmp1.png")
	im2.save("tmp2.png")
	result = xor_images(im1, im2)
	result.save('tmp.png')
	


