

_dictionary = set()
_dict_path = '/usr/share/dict/american-english'
def load_dict():
	for line in open(_dict_path).readlines():
		_dictionary.add(line.strip().lower())

def is_word(word):
	return word.lower() in _dictionary

load_dict()

