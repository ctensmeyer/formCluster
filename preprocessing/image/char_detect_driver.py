
import Image
import os
import sys
import char_detect_lib as char_lib
import datautils

IMAGE_EXT = '.png'

def main(image_dir, ocr_dir, out_dir):
	for f in os.listdir(image_dir):
		print f
		if not f.endswith(IMAGE_EXT):
			continue
		ocr_name = os.path.splitext(f)[0] + '.xml'
		in_image = os.path.join(image_dir, f)
		in_ocr = os.path.join(ocr_dir, ocr_name)
		out_image = os.path.join(out_dir, f)

		im = Image.open(in_image)
		im = im.convert('1')
		bounding_boxes = datautils.load_bounding_boxes(in_ocr)
		im = char_lib.char_detect(im, bounding_boxes)

		im.save(out_image)

if __name__ == "__main__":
	if len(sys.argv) < 4:
		print "[image_dir, ocr_dir, out_dir]"
		exit()
	image_dir = sys.argv[1] 
	ocr_dir = sys.argv[2] 
	out_dir = sys.argv[3]
	main(image_dir, ocr_dir, out_dir)

