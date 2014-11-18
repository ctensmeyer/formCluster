
import xml.etree.ElementTree as ET
import sys

_dictionary = set()
_dict_path = '/usr/share/dict/american-english'
def load_dict():
	for line in open(_dict_path).readlines():
		_dictionary.add(line.strip().lower())

def is_word(word):
	return word.lower() in _dictionary

load_dict()

def extract_characters(node):
	char_list = []
	for char in node.iter(tag='*'):
		if char.tag.endswith('charParams'):
			pos = (int(char.attrib['l']), int(char.attrib['t']))
			pos2 = (int(char.attrib['r']), int(char.attrib['b']))
			val = char.text
			char_ele = [val, pos, pos2]
			char_list.append(char_ele)
	return char_list


def has_dict_word(line, min_len=3):
	text = "".join(map(lambda c: c[0], line))
	for word in text.split():
		if len(word) >= min_len: #and is_word(word):
			return True
	return False
	
	
def extract_text_lines(ocr_path):
	tree = ET.parse(ocr_path)
	text_lines = []
	for node in tree.iter(tag='*'):
		if node.tag.endswith('line'):
			line = extract_characters(node)
			cleaned_line = clean_line(line)
			if has_dict_word(cleaned_line):
				text = "".join(map(lambda char: char[0], cleaned_line))
				pos1 = cleaned_line[0][1]
				pos2 = cleaned_line[-1][2]
				size = ( pos2[0] - pos1[0], pos2[1] - pos1[1] )
				text_lines.append( (text, pos1, size) )
	return text_lines


def clean_line(line):
	# remove chars
	for char in line:
		c = char[0]
		if ord(c) > 127 or not (c.isalpha() or c.isspace()):
			char[0] = ' '

	# trim space
	while line and line[0][0].isspace():
		line.pop(0)
	while line and line[-1][0].isspace():
		line.pop(-1)

	# condense spaces
	new_line = list()
	prev_space = False
	for x in xrange(len(line)):
		if not (prev_space and line[x][0].isspace()):
			new_line.append(line[x])
		prev_space = line[x][0].isspace()

	return new_line
	
if __name__ == "__main__":
	for line in extract_text_lines(sys.argv[1]):
		print line

