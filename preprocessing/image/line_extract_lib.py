
import Image
import ImageOps
import imageutils
import line_detect_lib as ld

LOW_THRESH = 40
HIGH_THRESH = 120

MIN_LINE_LENGTH = 0.01

def cc_size(cc, im_size):
	ul, lr = cc.bounding_box()
	rangex = lr[0] - ul[0]
	rangey = lr[1] - ul[1]

	if rangex > rangey:
		return rangex > (im_size[0] * MIN_LINE_LENGTH)
	else:
		return rangey > (im_size[1] * MIN_LINE_LENGTH)
	

def cc_contains_any(cc, points):
	s = set(cc.coords)
	for p in points:
		if p in s:
			return True
	return False

def line_extract(im):
	high_im = imageutils.binary(im, HIGH_THRESH)
	high_im = ImageOps.invert(high_im)
	high_im.save("high_im.pgm")
	high_cc_im, high_ccs = ld.find_ccs(high_im)
	print "Num ccs in high: ", len(high_ccs)
	map(lambda cc: cc.display(), high_ccs)
	high_ccs = filter(lambda cc: cc_size(cc, high_cc_im.size), high_ccs)
	print "Num ccs in high after filtering: ", len(high_ccs)
	map(lambda cc: cc.display(), high_ccs)
	cc_points = map(lambda cc: cc.coords[0], high_ccs)

	low_im = imageutils.binary(im, LOW_THRESH)
	low_im = ImageOps.invert(low_im)
	low_im.save("low_im.pgm")
	low_cc_im, low_ccs = ld.find_ccs(low_im)
	print "Num ccs in low: ", len(low_ccs)
	map(lambda cc: cc.display(), low_ccs)
	low_ccs = filter(lambda cc: cc_contains_any(cc, cc_points), low_ccs)
	print "Num ccs in low after filtering: ", len(low_ccs)
	map(lambda cc: cc.display(), low_ccs)

	return map(lambda cc: cc.bounding_box(), low_ccs)
	
