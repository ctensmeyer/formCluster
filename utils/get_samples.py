
import os
import sys
import shutil

ext = ".jpg"

def main(indir, outdir):
	try:
		os.makedirs(outdir)
	except:
		pass

	for sdir in os.listdir(indir):
		rdir = os.path.join(indir, sdir)
		for f in os.listdir(rdir):
			if f.endswith(ext):
				shutil.copy(os.path.join(rdir, f), os.path.join(outdir, sdir + ext))
				break

if __name__ == "__main__":
	indir = sys.argv[1]
	outdir = sys.argv[2]
	main(indir, outdir)

