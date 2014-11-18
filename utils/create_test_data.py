
import shutil
import random
import sys
import os



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

def get_filenames(in_dir):
	filenames = set()
	for _dir in os.listdir(in_dir):
		resolved = os.path.join(in_dir, _dir)
		for f in os.listdir(resolved):
			if f.endswith(".txt"):
				filenames.add(os.path.join(_dir, f))
	return filenames

def transfer(filenames, N, in_dir, out_dir):
	num_transfer = min(N, len(filenames))
	print "Moving %d documents" % num_transfer
	samples = random.sample(filenames, num_transfer)
	N = len(samples)
	for x, name in enumerate(samples):
		src_path = os.path.join(in_dir, name)
		dst_path = os.path.join(out_dir, name)
		shutil.copy(src_path, dst_path)
		if x % 10 == 0:
			print "%d documents have been moved, %2.1f%%" % (x, 100.0 * x / N)
	print "Done moving documents"
	

def main(in_dir, N, out_dir):
	filenames = get_filenames(in_dir)
	if N and out_dir:
		make_dirs(in_dir, out_dir)
		transfer(filenames, N, in_dir, out_dir)


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
	
