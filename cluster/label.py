
_label_file = "./labels.txt"
_label_dict = dict()
_loaded = False

prefixes = ['UK1911Census_EnglandWales_', 'UK1911_Census_', 'FormTypeDate_']

def strip_prefixes(s):
	l = 0
	while len(s) != l:
		l = len(s)
		for prefix in prefixes:
			if s.startswith(prefix):
				s = s[len(prefix):]
	return s
	

# load the label substitution table
def _load_check():
	global _loaded
	if _loaded:
		return
	for line in open(_label_file):
		line = line.strip()
		if not line or line.startswith('#'):
			continue
		tokens = line.split()
		_label_dict[strip_prefixes(tokens[0])] = strip_prefixes(tokens[1])
	_loaded = True

def preprocess_label(label):
	_load_check()
	label = strip_prefixes(label)
	while label in _label_dict:
		label = _label_dict[label]
	return label

	
