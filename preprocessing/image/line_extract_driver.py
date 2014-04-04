import Image
import ImageDraw
import os
import sys
import line_extract_lib as line_lib

IMAGE_EXT = '_linesH.pgm'
LINE_EXT = '_endpoints.xml'

def main(in_dir, out_dir):
	try:
		os.makedirs(out_dir)
	except:
		pass
	for f in os.listdir(in_dir):
		if not f.endswith(IMAGE_EXT):
			continue
		print f
		h_name = f
		v_name = f.replace("linesH", "linesV")
		orig_name = f.replace(IMAGE_EXT, ".jpg")
		ocr_name = f.replace(IMAGE_EXT, ".xml")
		save_name = f.replace(IMAGE_EXT, "_with_lines.jpg")

		h_path = os.path.join(in_dir, h_name)
		v_path = os.path.join(in_dir, v_name)
		orig_path = os.path.join(in_dir, orig_name)
		ocr_path = os.path.join(in_dir, ocr_name)
		save_path = os.path.join(out_dir, save_name)
		out_path = os.path.join(out_dir, f.replace(IMAGE_EXT, "")  + LINE_EXT)
		verify_path = os.path.join(out_dir, f.replace(IMAGE_EXT, "")  + "_verify.png")

		h_im = Image.open(h_path)
		v_im = Image.open(v_path)
		#if os.path.exists(ocr_path):
		#	line_lib.remove_ocr_chars(h_im, ocr_path)
		#	line_lib.remove_ocr_chars(v_im, ocr_path)
		#	h_im.save("h_removed.pgm")
		#	v_im.save("v_removed.pgm")
		#else:
		#	print "Cannot find ", ocr_path
		#orig_im = Image.open(orig_path).convert("RGB")
		#verify_im = Image.new("RGB", h_im.size, "white")

		verify_im = line_lib.write_line_file(h_im, v_im, out_path, get_image=True)
		verify_im.save(verify_path)


if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "[in_dir, out_dir]"
		exit()
	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	main(in_dir, out_dir)

