
import Image
import sys
import os
import char_detect_lib as cl 


IMAGE_EXT = '.png'

def main(char_dir, line_dir, out_dir):
	for f in os.listdir(char_dir):
		if not f.endswith(IMAGE_EXT):
			continue
		print f
		char_file = os.path.join(char_dir, f)
		line_file = os.path.join(line_dir, f)
		out_file = os.path.join(out_dir, f)
		try:
			im_char = Image.open(char_file)
			im_line = Image.open(line_file)
		except:
			print "Filename %s couldn't be opened in both directories" % f
			continue

		result = cl.merge_with_line(im_char, im_line)
		result.save(out_file)
	
if __name__ == "__main__":
	if len(sys.argv) < 4:
		print"[char_dir line_dir out_dir]"
		exit()
	char_dir = sys.argv[1]
	line_dir = sys.argv[2]
	out_dir = sys.argv[3]
	main(char_dir, line_dir, out_dir)

