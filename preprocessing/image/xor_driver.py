
import Image
import sys
import os
import char_detect_lib as cl

IMAGE_EXT = '.png'

def main(in_dir, out_dir):
	for f1 in os.listdir(in_dir):
		if not f1.endswith(IMAGE_EXT):
			continue
		print f1
		im1_file = os.path.join(in_dir, f1)
		im1 = Image.open(im1_file)
		for f2 in os.listdir(in_dir):
			if not f2.endswith(IMAGE_EXT) or f2 > f1:
				continue
			print "\t", f2
			im2_file = os.path.join(in_dir, f2)
			im2 = Image.open(im2_file)
			out_file = os.path.join(out_dir, "%s_%s.png" % (os.path.splitext(f1)[0],
											os.path.splitext(f2)[0]))
			result = cl.compare(im1, im2)
			result.save(out_file)


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "[indir outdir]"
		exit()
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	main(in_dir, out_dir)

