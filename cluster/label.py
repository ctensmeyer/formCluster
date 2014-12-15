
_label_file = "./labels.txt"
_label_dict = dict()
_loaded = False

prefixes = ['UK1911Census_EnglandWales_', 'UK1911_Census_', 'FormTypeDate_']

# load the label substitution table
def _load_check():
	if _loaded:
		return
	for line in open(_label_file):
		line = line.strip()
		if line.startswith('#'):
			continue
		tokens = line.split()
		_label_dict[tokens[0]] = tokens[1]

def preprocess_label(label):
	_load_check()
	for prefix in prefixes:
		if label.startswith(prefix):
			label = label[len(prefix):]
	if label in _label_dict:
		return _label_dict[label]
	else:
		return label
	
