
import Image
import os
import sys
import line_detect_lib as line_lib

IMAGE_EXT = '.png'

def main(in_dir, out_dir):
	for f in os.listdir(in_dir):
		if not f.endswith(IMAGE_EXT):
			continue
		print f
		in_image = os.path.join(in_dir, f)
		out_image = os.path.join(out_dir, f)

		im = Image.open(in_image)
		im = line_lib.line_detect(im)

		im.save(out_image)

if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "[in_dir, out_dir]"
		exit()
	in_dir = sys.argv[1]  # image must be greyscale binary for now
	out_dir = sys.argv[2]
	main(in_dir, out_dir)

