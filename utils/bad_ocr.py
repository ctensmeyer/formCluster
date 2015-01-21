
# This script will print out the ids of the forms with a relatively small OCR file,
#  usually indicating that the transcription is partial.  $bad_mult controls how
#  small a file has to be compared to the median file size in order to be counted as bad

import os
import sys

bad_mult = 0.5

def main(indir):
	for sdir in os.listdir(indir):
		rdir = os.path.join(indir, sdir)
		filenames = list()
		for f in os.listdir(rdir):
			if f.endswith(".jpg"):
				filenames.append(os.path.join(rdir, os.path.splitext(f)[0] + ".xml"))
		sizes = map(lambda f: os.path.getsize(f), filenames)
		sizes.sort()
		median_size = sizes[len(sizes) / 2]
		#print rdir, median_size
		for f in filenames:
			if os.path.getsize(f) < median_size * bad_mult:
				print os.path.splitext(os.path.basename(f))[0]
				#print os.path.join(sdir, os.path.basename(f))


if __name__ == "__main__":
	indir = sys.argv[1]
	main(indir)

