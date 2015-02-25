
import os
import sys

img_ext = ".txt"
expected_ext = [".jpg"]


def main(indir):
	for sdir in os.listdir(indir):
		rdir = os.path.join(indir, sdir)
		ids = list()
		for f in os.listdir(rdir):
			if f.endswith(img_ext):
				ids.append(os.path.splitext(f)[0])
		for _id in ids:
			for ext in expected_ext:
				f = _id + ext
				if not os.path.exists(os.path.join(rdir, f)):
					print os.path.join(sdir, _id) + " is missing " + f



if __name__ == "__main__":
	indir = sys.argv[1]
	main(indir)

