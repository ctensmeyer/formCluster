
import os
import sys
import xml.etree.ElementTree as ET
import json
import Image
import ImageDraw


def find_boxes(root):
	for child in root:
		if child.tag.endswith('page'):
			page = child
	
	boxes = []
	
	for x, child in enumerate(page):
		#print child.tag
		if child.tag.endswith('block'):
			#print child.attrib
			ul = (int(child.get('l')), int(child.get('t')))
			br = (int(child.get('r')), int(child.get('b')))
			box = (ul, br)
			boxes.append(box)
			
	return boxes

def main(dirname, outputdir):

	global_overlap = 0
	global_area = 0
	global_num = 0
	global_num_overlaps = 0
	for filename in os.listdir(dirname):
		if not filename.endswith(".xml"):
			continue
		tree = ET.parse(os.path.join(dirname, filename))
		root = tree.getroot()

		boxes = find_boxes(root)
		br = (max([x[1][0] for x in boxes]), max([x[1][1] for x in boxes]))

		overlaps = {idx: {'total': 0} for idx in xrange(len(boxes))}
		for x, box1 in enumerate(boxes):
			for y, box2 in enumerate(boxes):
				if x == y:
					continue
				area = overlap(box1, box2)
				if area:
					overlaps[x][y] = area
					overlaps[x]['total'] += area

		if outputdir:
			im = Image.new("RGB", br, "white")
			draw = ImageDraw.Draw(im)
			for box in boxes:
				draw.rectangle(box, outline="black")	

			outfilename = os.path.join(outputdir, os.path.splitext(filename)[0] + "_regions.jpeg")
			print outfilename
			im.save(outfilename)
			#im.show()
		
		#print json.dumps(overlaps, indent=4)
		total_overlap = 0
		num_overlapping = 0
		for dct in overlaps.values():
			total_overlap += dct['total']
			if dct['total'] != 0:
				num_overlapping += 1

		total_overlap /= 2
		print "Filename: ", filename
		print "Total Overlap area: ", total_overlap
		print "Percent Overlap area: %.2f%%" % (100 * total_overlap / float(br[0] * br[1]))
		print "Num Blocks Overlapping: ", num_overlapping
		print "Percent Blocks Overlapping: %.2f%%" % (100 * num_overlapping / float(len(boxes)))
		print

		global_overlap += total_overlap
		global_area += br[0] * br[1]
		global_num += len(boxes)
		global_num_overlaps += num_overlapping
		
	print "Global Overlap area: ", global_overlap
	print "Global Percent Overlap area: %.2f%%" % (100 * global_overlap / float(global_area))
	print "Global Num Blocks Overlapping: ", global_num
	print "Global Percent Blocks Overlapping: %.f%%" % (100 * global_num_overlaps / float(global_num))
	print
			

def overlap(box1, box2):
	x11 = box1[0][0]
	x12 = box1[1][0]
	x21 = box2[0][0]
	x22 = box2[1][0]

	x_overlap = max(0, min(x12, x22) - max(x11, x21))


	y11 = box1[0][1]
	y12 = box1[1][1]
	y21 = box2[0][1]
	y22 = box2[1][1]

	y_overlap = max(0, min(y12, y22) - max(y11, y21))

	return x_overlap * y_overlap

if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise Exception("Too few args")
	dirname = sys.argv[1]
	outputdir = sys.argv[2] if len(sys.argv) > 2 else None
	main(dirname, outputdir)



