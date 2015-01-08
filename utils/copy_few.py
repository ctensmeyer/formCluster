
# This script will copy over the files associted with every entry in missing_file
#   from the fromdir to the todir.
# Used to correct some mistakes when copying/processing a large dataset

import os
import sys
import shutil

expected_ext = [".xml", "_linesH.png", "_linesV.png"]


def main(missing_file, fromdir, todir):
	for line in open(missing_file).readlines():
		line = line.strip()
		tokens = line.split('/')

		# This part is to copy over all the files related to each missing entry
		try:
			os.makedirs(os.path.join(todir, tokens[0]))
		except:
			pass
		shutil.copy(os.path.join(fromdir, tokens[0], 'Lines', tokens[1] + '_linesH.pgm'),
					os.path.join(todir, line + '_linesH.pgm'))
		shutil.copy(os.path.join(fromdir, tokens[0], 'Lines', tokens[1] + '_linesV.pgm'),
					os.path.join(todir, line + '_linesV.pgm'))
		shutil.copy(os.path.join(fromdir, tokens[0], 'OCR', tokens[1] + '.xml'),
					os.path.join(todir, line + '.xml'))
		shutil.copy(os.path.join(fromdir, tokens[0], 'OCR', tokens[1] + '.txt'),
					os.path.join(todir, line + '.txt'))
		shutil.copy(os.path.join(fromdir, tokens[0], 'Data', tokens[1] + '.txt.gz'),
					os.path.join(todir, line + '.txt.gz'))

		## This part removes everything associated with that file in the fromdir
		#try:
		#	os.remove(os.path.join(fromdir, line + '_linesH.pgm'))
		#except:
		#	pass
		#try:
		#	os.remove(os.path.join(fromdir, line + '_linesH.png'))
		#except:
		#	pass
		#try:
		#	os.remove(os.path.join(fromdir, line + '_linesV.pgm'))
		#except:
		#	pass
		#try:
		#	os.remove(os.path.join(fromdir, line + '_linesV.png'))
		#except:
		#	pass
		#try:
		#	os.remove(os.path.join(fromdir, line + '.xml'))
		#except:
		#	pass
		#try:
		#	os.remove(os.path.join(fromdir, line + '.jpg'))
		#except:
		#	pass
		#try:
		#	os.remove(os.path.join(fromdir, line + '.txt'))
		#except:
		#	pass
		##try:
		##	os.remove(os.path.join(fromdir, tokens[0],  'Data', tokens[1]  + '.txt.gz'))
		##except:
		##	pass


if __name__ == "__main__":
	missing_file = sys.argv[1]
	fromdir = sys.argv[2]
	todir = sys.argv[3]
	main(missing_file, fromdir, todir)

