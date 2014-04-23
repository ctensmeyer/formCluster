
import shutil
import random
import sys
import os


#extensions = [".xml", "_line.xml", "_FormType.txt", ".jpg", "_linesH.pgm", "_linesV.pgm"]
extensions = [".xml", "_line.xml", "_FormType.txt", ".jpg", "_endpoints.xml","_verify.png"]
ignore_extensions = (".db","_linesH.pgm", "_linesV.pgm")


def make_dirs(in_dir, out_dir):
	try:
		os.mkdir(out_dir)
	except:
		pass
	for _dir in os.listdir(in_dir):
		resolved = os.path.join(in_dir, _dir)
		if os.path.isdir(resolved):
			try:
				os.mkdir(os.path.join(out_dir, _dir))
			except:
				pass

def get_basename(filename):
	base = ""
	for ext in extensions:
		if filename.endswith(ext):
			base = filename[:-len(ext)]
	return base

def get_basenames(in_dir):
	has_all = set()
	lacks = set()
	for _dir in os.listdir(in_dir):
		resolved = os.path.join(in_dir, _dir)
		for f in os.listdir(resolved):
			if f.endswith(ignore_extensions):
				continue
			basename = get_basename(f)
			if not basename:
				print "Could not find basename of: ", os.path.join(resolved, f)
				continue
			if all(map(lambda ext: os.path.exists(os.path.join(resolved, basename + ext)), extensions)):
				has_all.add(os.path.join(_dir, basename))
			else:
				lacks.add(basename)
	print "%d documents total" % (len(has_all) + len(lacks))
	print "%d documents have all parts" % len(has_all)
	print "%d images do not have all parts" % len(lacks)
	return has_all

def transfer(basenames, N, in_dir, out_dir):
	num_transfer = min(N, len(basenames))
	print "Moving %d documents" % num_transfer
	samples = random.sample(basenames, num_transfer)
	N = len(samples)
	for x, name in enumerate(samples):
		for ext in extensions:
			src_path = os.path.join(in_dir, name + ext)
			dst_path = os.path.join(out_dir, name + ext)
			shutil.copy(src_path, dst_path)
		if x % 10 == 0:
			print "%d documents have been moved, %2.1f%%" % (x, 100.0 * x / N)
	print "Done moving documents"
	

def main(in_dir, N, out_dir):
	basenames = get_basenames(in_dir)
	if N and out_dir:
		make_dirs(in_dir, out_dir)
		transfer(basenames, N, in_dir, out_dir)

	exit()
	count = 0
	for basename in basenames:
		print basename
		count += 1
		if count > 100:
			break


if __name__ == "__main__":
	if len(sys.argv) <= 1 or sys.argv[1] in ["-h", "-help"]:
		print "[in_dir] N [out_dir]"
		exit()
	in_dir = sys.argv[1]
	N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
	out_dir = sys.argv[3] if len(sys.argv) > 3 else None
	if N and out_dir:
		print "Moving at most %d documents from %s to %s" % (N, in_dir, out_dir)
	else:
		print "Counting documents in ", in_dir
		
	main(in_dir, N, out_dir)
	
