
import ocr
import Image
import ImageOps
import ImageDraw
import imageutils
import line_detect_lib as ld

LOW_THRESH = 30
HIGH_THRESH = 120

MIN_LINE_LENGTH = 0.01

def fill_bbs(im, bbs, color=255):
	draw = ImageDraw.Draw(im)
	for bb in bbs:
		draw.rectangle(bb, fill=color)


def remove_ocr_chars(im, ocr_path):
	bbs = ocr.get_bounding_boxes(ocr_path)
	fill_bbs(im, bbs, color=0)

def cc_size(cc, im_size):
	ul, lr = cc.bounding_box
	rangex = lr[0] - ul[0]
	rangey = lr[1] - ul[1]

	if rangex > rangey:
		return rangex > (im_size[0] * MIN_LINE_LENGTH)
	else:
		return rangey > (im_size[1] * MIN_LINE_LENGTH)

def extract_line_from_cc(cc):
	cc.make_mask()
	mask = cc.mask
	line_len = 0
	line_thick = 0
	line_center = 0
	line_start = 0
	line_orientation = ""
	size = mask.size
	if size[0] > size[1]:
		line_orientation = "Horizontal"
	else:
		line_orientation = "Vertical"
		mask = mask.rotate(90)
		size = mask.size
	line_len = size[0]
	pix = mask.load()
	mass_pos = 0
	thicknesses = list()
	for x in xrange(size[0]):
		thickness = 0
		for y in xrange(size[1]):
			if pix[x, y]:
				thickness += 1
				mass_pos += y
			thicknesses.append(thickness)
	line_thick = int(round(sum(thicknesses) / float(len(thicknesses))))
	rel_c_of_mass = int(round(mass_pos / float(len(cc.coords))))
	ul, lr = cc.bounding_box
	if line_orientation == "Horizontal":
		line_center = ul[1] + rel_c_of_mass
		line_start = ul[0]
	else:
		line_center = ul[0] + rel_c_of_mass
		line_start = ul[1]
	
	return line_orientation, line_len, line_thick, line_center, line_start
		
def cc_contains_any(cc, points):
	s = set(cc.coords)
	for p in points:
		if p in s:
			return True
	return False

def get_line_ccs(im):
	high_im = imageutils.binary(im, HIGH_THRESH)
	high_im = ImageOps.invert(high_im)
	#high_im.save("high_im.pgm")
	high_cc_im, high_ccs = ld.find_ccs(high_im)
	#print "Num ccs in high: ", len(high_ccs)
	#map(lambda cc: cc.display(), high_ccs)
	high_ccs = filter(lambda cc: cc_size(cc, high_cc_im.size), high_ccs)
	#print "Num ccs in high after filtering: ", len(high_ccs)
	#map(lambda cc: cc.display(), high_ccs)
	cc_points = map(lambda cc: cc.coords[0], high_ccs)

	low_im = imageutils.binary(im, LOW_THRESH)
	low_im = ImageOps.invert(low_im)
	#low_im.save("low_im.pgm")
	low_cc_im, low_ccs = ld.find_ccs(low_im)
	#print "Num ccs in low: ", len(low_ccs)
	#map(lambda cc: cc.display(), low_ccs)
	low_ccs = filter(lambda cc: cc_contains_any(cc, cc_points), low_ccs)
	#print "Num ccs in low after filtering: ", len(low_ccs)
	#map(lambda cc: cc.display(), low_ccs)

	return low_ccs

def write_line_file(horz_im, vert_im, outfile, get_image=False):
	horz_ccs = get_line_ccs(horz_im)
	vert_ccs = get_line_ccs(vert_im)
	horz_lines = map(extract_line_from_cc, horz_ccs)
	vert_lines = map(extract_line_from_cc, vert_ccs)
	with open(outfile, 'w') as out:
		out.write("<Lines>\n")
		out.write("\t<HorizontalLines>\n")
		for o, l, t, c, s in horz_lines:
			assert o == "Horizontal"
			out.write("\t\t<Line x=\"%s\" y=\"%s\" length=\"%s\" thickness=\"%s\"/>\n" % (s, c, l, t) )
		out.write("\t</HorizontalLines>\n")
		out.write("\t<VerticalLines>\n")
		for o, l, t, c, s in vert_lines:
			assert o == "Vertical"
			out.write("\t\t<Line x=\"%s\" y=\"%s\" length=\"%s\" thickness=\"%s\"/>\n" % (c, s, l, t) )
		out.write("\t</VerticalLines>\n")
		out.write("</Lines>\n")
	if get_image:
		im = Image.new("RGB", horz_im.size, "white")
		draw_lines(horz_lines, vert_lines, im)
		return im

def draw_lines(horz_lines, vert_lines, im):
	draw = ImageDraw.Draw(im)
	for o, l, t, c, s in horz_lines:
		x, y = s, c
		y -= t / 2
		draw.line( (x, y, x + l, y), width=t * 3, fill='red')
	for o, l, t, c, s in vert_lines:
		x, y = c, s
		x -= t / 2
		draw.line( (x, y, x, y + l), width=t * 3, fill='blue')
	
