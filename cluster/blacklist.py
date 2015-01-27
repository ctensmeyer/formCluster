
_blacklist = set()

_black_list_filename = "/home/chris/formCluster/cluster/blacklist.txt"
_loaded = False

def _load_check():
	global _loaded
	if _loaded:
		return
	for line in open(_black_list_filename).readlines():
		line = line.strip()
		if not line or line.startswith('#'):
			continue
		_blacklist.add(line)
	_loaded = True

def contains(basename):
	_load_check()
	return basename in _blacklist
	
	
