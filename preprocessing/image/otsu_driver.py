
import Image
import imageutils
import os
import sys

IMAGE_EXT = '.png'


def main(indir, outdir):
	for f in os.listdir(indir):
		if not f.endswith(IMAGE_EXT):
			continue

		print f
		inimage = os.path.join(indir, f)
		im = Image.open(inimage)
		im = im.convert('L')
		#threshold = imageutils.global_otsu(im)
		#print "threshold:", threshold
		#im = imageutils.binary(im, threshold)
		im = imageutils.local_otsu(im, 4, 4)

		outfile = os.path.join(outdir, os.path.basename(inimage))
		im.save(outfile)
	imageutils.display_all_thresh()

if __name__ == "__main__":
	if len(sys.argv) < 3:
		raise Exception("[indir outdir]")
	indir = sys.argv[1]
	outdir = sys.argv[2]
	main(indir, outdir)

