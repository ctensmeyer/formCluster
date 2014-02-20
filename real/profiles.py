
import xml.etree.ElementTree as ET

_test_path = "/home/chris/Ancestry/Data/large/1911Wales/UnClassified/rg14_31687_0058_06_line.xml"

def extract_profiles(prof_path):
	tree = ET.parse(prof_path)
	root = tree.getroot()
	ele = root.find('LineProfiles')
	profiles = {}
	for child in ele:
		profile = []
		for score in child:
			profile.append(float(score.text))
		profiles[child.tag] = profile
	return profiles

def get_size(prof_path):
	tree = ET.parse(prof_path)
	root = tree.getroot()
    raise Exception("Not quite done yet!")
    

if __name__ == "__main__":
	extract_profiles(_test_path)

