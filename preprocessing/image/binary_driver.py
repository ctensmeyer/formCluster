
import Image
import ImageEnhance
import os
import sys
import imageutils



def main(indir, outdir, thresh):
	try:
		os.makedirs(outdir)
	except:
		pass
	for f in os.listdir(indir):
		if not f.lower().endswith('.jpg'):
			continue
		infile = os.path.join(indir, f)
		im = Image.open(infile)
		im = imageutils.greyscale(im)
		im = imageutils.binary(im, thresh)
		outfile = os.path.join(outdir, f)
		im.save(outfile)


if __name__ == "__main__":
	if len(sys.argv) < 4:
		raise Exception("[indir outdir threshold]")
	indir = sys.argv[1]
	outdir = sys.argv[2]
	thresh = int(sys.argv[3])
	main(indir, outdir, thresh)

