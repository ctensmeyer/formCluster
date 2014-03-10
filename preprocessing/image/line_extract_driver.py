import Image
import ImageDraw
import os
import sys
import line_extract_lib as line_lib

IMAGE_EXT = '.pgm'

def main(in_dir, out_dir):
	try:
		os.makedirs(out_dir)
	except:
		pass
	for f in os.listdir(in_dir):
		if not f.endswith(IMAGE_EXT):
			continue
		print f
		in_image = os.path.join(in_dir, f)
		out_image = os.path.join(out_dir, f)

		im = Image.open(in_image)
		lines = line_lib.line_extract(im)
		im = im.convert("RGB")
		draw = ImageDraw.Draw(im)

		for line in lines:
			draw.line(line, fill="red", width=10)

		im.save(out_image)


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "[in_dir, out_dir]"
		exit()
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	main(in_dir, out_dir)

