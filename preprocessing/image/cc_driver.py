
import Image
import sys
import os
import line_detect_lib as line_detect


def main(indir, outdir):
	for f in os.listdir(indir):
		if not f.endswith('.png'):
			continue
		print f
		inimage = os.path.join(indir, f)
		outimage = os.path.join(outdir, f)
		im = Image.open(inimage)
		im = im.convert('1')
		out = line_detect.find_ccs(im)[0]
		colors = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (127, 127, 0),
					  (127, 0, 127), (0, 127, 127), (200, 100, 100), (100, 200, 100),
					  (100, 100, 200), (200, 50, 50), (50, 200, 50), (50, 50, 200),
					  (127, 0, 0), (0, 127, 0), (0, 0, 127) ]
		out = line_detect.color_components(im, out, colors)
		out.save(outimage)


if __name__ == "__main__":
	if len(sys.argv) < 3:
		raise Exception("[indir outdir]")
	indir = sys.argv[1]
	outdir = sys.argv[2]
	main(indir, outdir)

