
import Image
import sys
import imageutils



def main(infile, outfile, window, value_sig, space_sig):
	im = Image.open(infile)
	im = imageutils.greyscale(im)
	old = im.size
	new_width = im.size[0] / 5
	new_height = im.size[1] / 5
	
	im = im.resize((new_width, new_height))
	im = imageutils.bilateral(im, window, value_sig, space_sig)
	im = im.resize(old)
	im.save(outfile)


if __name__ == "__main__":
	if len(sys.argv) < 5:
		print "[infile] [outfile] [window size] [value sigma] [spacial sigma]"
		exit()
	infile = sys.argv[1]
	outfile = sys.argv[2]
	window = int(sys.argv[3])
	value_sig = float(sys.argv[4])
	space_sig = float(sys.argv[5])
	main(infile, outfile, window, value_sig, space_sig)

