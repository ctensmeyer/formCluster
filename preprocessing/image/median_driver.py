
import Image
import imageutils
import sys


def main(input_image, output_image):
	print input_image
	im = Image.open(input_image)
	im = imageutils.median_filter(im, 15, 3)
	im.save(output_image)
	print 'done'


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "[input_image output_image]"
		exit()
	input_image = sys.argv[1]
	output_image = sys.argv[2]
	main(input_image, output_image)

