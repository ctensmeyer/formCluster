
import xml.etree.ElementTree as ET
from components import TextLine, Char


def extract_characters(node):
	char_list = []
	for char in node.iter(tag='*'):
		if char.tag.endswith('charParams'):
			pos = (int(char.attrib['l']), int(char.attrib['t']))
			pos2 = (int(char.attrib['r']), int(char.attrib['b']))
			#attribs = {key: int(val) for (key, val) in char.attrib.iteritems()}
			val = char.text
			char_ele = Char(val, pos, pos2)
			char_list.append(char_ele)
	return char_list
	
	
def extract_text_lines(ocr_path):
	tree = ET.parse(ocr_path)
	text_lines = []
	for node in tree.iter(tag='*'):
		if node.tag.endswith('line'):
			char_list = extract_characters(node)
			l = int(node.get('l'))
			t = int(node.get('t'))
			r = int(node.get('r'))
			b = int(node.get('b'))
			position = (l, t)
			size = (r - l, b - t)
			text_lines.append(TextLine(char_list, position, size))
	return text_lines

def clean_lines(lines):
	filtered_lines = list()
	for line in lines:
		line.filter_nonalpha()
		line.trim()
		line.condense_space()
		if line.has_dict_word():
			filtered_lines.append(line)
	return filtered_lines
	


_test_file = "/home/chris/Ancestry/Data/test/UnClassified/rg14_31687_0058_06.xml"
if __name__ == "__main__":
	lines = extract_text_lines(_test_file)
	filtered_lines = list()
	for line in lines:
		print line
		line.filter_nonalpha()
		line.trim()
		print line
		if line.has_dict_word():
			filtered_lines.append(line)

	print "Filtered"
	for line in filtered_lines:
		print line
	

