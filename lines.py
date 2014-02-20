
import os
import sys
import math
import json
import xml.etree.ElementTree as ET
import Image, ImageDraw



def find_lines(root):
	segments = root[0]
	lines = []
	for line in segments:
		p1 = (int(line.get('X1')), int(line.get('Y1')))
		p2 = (int(line.get('X2')), int(line.get('Y2')))
		lines.append( (p1, p2) )

	return lines


def find_intersections(lines):
	intersections = []

	for line1 in lines:
		for line2 in lines:
			if line1 is line2 or (line2, line1) in intersections:
				continue
			if intersect(line1, line2):
				intersections.append( (line1, line2) )

	return intersections

def intersect(line1, line2, close=False):
	# return point of intersection or False
	A1, B1, C1 = get_equation(line1)	
	A2, B2, C2 = get_equation(line2)	
	det = A1 * B2 - A2 * B1
	if det == 0:
		return False
	x = (B2 * C1 - B1 * C2) / det
	y = (A1 * C2 - A2 * C1) / det

	if close:
		for p1 in line1:
			for p2 in line2:
				if dist(p1, p2) < 5:
					return True

	return (x, y) if is_in_box((x,y), line1) and is_in_box((x,y), line2) else False


def build_graph(lines, intersections):
	# Return (V={idx : {metadata}}, E={idx : {idx: {recursive}} )
	tmp = {line : x for x, line in enumerate(lines)}
	V = {idx: {'line': lines[idx], 'len': line_len(lines[idx]), 'num_edges': 0} for idx in xrange(len(lines))}
	E = {idx: {} for idx in xrange(len(lines))}

	for line1, line2 in intersections:
		idx1 = tmp[line1]
		idx2 = tmp[line2]
		V[idx1]['num_edges'] += 1
		V[idx2]['num_edges'] += 1
		E[idx1][idx2] = E[idx2]
		E[idx2][idx1] = E[idx1]

	return (V, E)


def is_in_box(pt, box):
	# lines and boxes look the same
	return (pt[0] >= min(box[0][0], box[1][0]) and pt[0] <= max(box[0][0], box[1][0]) and
			  pt[1] >= min(box[0][1], box[1][1]) and pt[1] <= max(box[0][1], box[1][1]) )
	

def get_equation(line):
	# Ax + By = C
	A = line[1][1] - line[0][1]
	B = line[1][0] - line[0][0]
	C = A * line[0][0] + B * line[0][1]
	return (A, B, C)


def line_len(line):
	x2 = (line[0][0] - line[1][0]) ** 2
	y2 = (line[0][1] - line[1][1]) ** 2

	return math.sqrt(x2 + y2)

def dist(p1, p2):
	return line_len((p1, p2))

def draw_image(filename, lines, outputdir, inputdir, br):
	filepart = os.path.splitext(filename)[0]
	filepath = os.path.join(inputdir, filepart + ".jpg")
	if os.path.isfile(filepath):
		im = Image.open(filepath)
	else:
		im = Image.new("RGB", br, "white")
	draw = ImageDraw.Draw(im)
	for line in lines:
		draw.line(line, fill="green", width=5)	

	outfilename = os.path.join(outputdir, filepart + "_lines.jpeg")
	#print outfilename
	im.save(outfilename)
	#im.show()

def split_lines(lines, intersections):
	new_lines = []
	for x, line in enumerate(lines):
		inters = [i for i in intersections if i[0] == line or i[1] == line]
		points = map(lambda tup: intersect(tup[0], tup[1]), inters)
		points = filter(lambda p: p != line[0] and p != line[1], points)
		points = sorted(points)
		if not points:
			continue
		prev = line[0]
		for point in points:
			new_lines.append( (prev, point) )
			prev = point
		new_lines.append( (prev, line[1]) )
	return new_lines, find_intersections(new_lines)
		

def main(dirname, outputdir, inputdir, outfile):
	
	edge_counts = {}

	for filename in os.listdir(dirname):
		if not filename.endswith(".xml"):
			continue
		print filename
		tree = ET.parse(os.path.join(dirname, filename))
		root = tree.getroot()
		br = (int(root.get('ImageWidth')), int(root.get('ImageHeight')))

		lines = find_lines(root)
		intersections = find_intersections(lines)
		lines, intersections = split_lines(lines, intersections)
		if outputdir:
			draw_image(filename, lines, outputdir, inputdir, br)

		#print len(intersections)
		(V, E) = build_graph(lines, intersections)
		#print json.dumps(V, indent=4)
		#for idx, edges in E.iteritems():
		#	print idx, ":", sorted(edges.keys())
		edge_count = sorted([V[x]['num_edges'] for x in xrange(len(V))], reverse=True)
		box_count = num_boxes(E)
		print
		print "%s: box_count: %d" % (filename, box_count)
		print edge_count
		edge_counts[filename] = edge_count

	name_id = {filename:x for x,filename in enumerate(edge_counts.keys())}
	id_name = {x:filename for x,filename in enumerate(edge_counts.keys())}
	sim_matrix = list()
	for filename in name_id:
		row = list()
		for filename2 in name_id:
			sim = similarity(edge_counts[filename], edge_counts[filename2])
			row.append(sim)
		sim_matrix.append(row)
				
	if outfile:
		write_mat(sim_matrix, id_name, outfile)

def write_mat(sim_matrix, id_name, outfile):
	out = open(outfile, 'w')
	for x, row in enumerate(sim_matrix):
		out.write("%s %s %s\n" % (id_name[x], x, row))
	out.flush()
	out.close()

# lower numbers are better
def similarity(edge_counts1, edge_counts2):
	if len(edge_counts1) < len(edge_counts2):
		edge_counts1, edge_counts2 = (edge_counts2, edge_counts1)

	similarity = 0
	for x in xrange(len(edge_counts2)):
		similarity += abs(edge_counts1[x] - edge_counts2[x])

	# extra penalty for different number of nodes
	for x in xrange(len(edge_counts2), len(edge_counts1)):
		similarity += edge_counts1[x] + 1

	return similarity


def num_faces(edge_counts):
	verts = len(edge_counts)
	edges = reduce(lambda x,y: x+y, edge_counts) / 2
	# Euler's formula for planer graphs
	faces = 2 + edges - verts
	return faces

def num_boxes(E):
	num = 0
	for v1 in E:
		for v2 in E:
			# same vertex or adjacent
			if v1 <= v2 or v2 in E[v1].keys():
				continue
			intersection = [x for x in E[v1].keys() if x in E[v2].keys()]
			if len(intersection) > 1:
				num += 1
	return num
	
	

if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise Exception("Too few args")
	dirname = sys.argv[1]
	outputdir = sys.argv[2] if len(sys.argv) > 2 else None
	inputdir = sys.argv[3] if len(sys.argv) > 3 else None
	outfile = sys.argv[4] if len(sys.argv) > 4 else None
	main(dirname, outputdir, inputdir, outfile)

