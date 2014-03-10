
import Image
import ImageEnhance
import os
import sys
import imageutils

IMAGE_EXT = '.pgm'


def main(indir, outdir):#, thresh):
	for f in os.listdir(indir):
		if not f.lower().endswith(IMAGE_EXT):
			continue
		image_dir = os.path.join(outdir, f)
		try:
			os.makedirs(image_dir)
		except:
			pass
		for thresh in xrange(0, 260, 10):
			infile = os.path.join(indir, f)
			im = Image.open(infile)
			im = imageutils.greyscale(im)
			im = imageutils.binary(im, thresh)
			outfile = os.path.join(image_dir, "%d_%s" % (thresh, f))
			im.save(outfile)


if __name__ == "__main__":
	#if len(sys.argv) < 4:
	if len(sys.argv) < 3:
		raise Exception("[indir outdir threshold]")
	indir = sys.argv[1]
	outdir = sys.argv[2]
	#thresh = int(sys.argv[3])
	main(indir, outdir)#, thresh)

