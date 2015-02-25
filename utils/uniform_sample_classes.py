
# Does the actual work in creating subsets of data.  See subsets.sh

import shutil
import random
import sys
import os
import collections




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
	filenames = collections.defaultdict(list)
	for _dir in os.listdir(in_dir):
		resolved = os.path.join(in_dir, _dir)
		for f in os.listdir(resolved):
			if f.endswith(".txt"):
				filenames[_dir].append(os.path.join(_dir, f))
	return filenames

def transfer(filenames, N, in_dir, out_dir):
	num_to_transfer = sum(map(lambda l: min(len(l), N), filenames.values()))
	print "Copying %d documents" % num_to_transfer
	num_transfered = 0
	for _dir in filenames:
		files = filenames[_dir]
		if len(files) <= N:
			to_transfer = files
		else:
			to_transfer = random.sample(files, N)
		for name in to_transfer:
			
			# features
			src_path = os.path.join(in_dir, name)
			dst_path = os.path.join(out_dir, name)
			shutil.copy(src_path, dst_path)

			# image
			src_path = os.path.join(in_dir, name[:-4] + ".jpg")
			dst_path = os.path.join(out_dir, name[:-4] + ".jpg")
			shutil.copy(src_path, dst_path)

			num_transfered += 1
			if num_transfered % 10 == 0:
				print "%d documents have been copied, %2.1f%%" % (
					num_transfered, 100.0 * num_transfered / num_to_transfer)

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
		print "Moving at most N %d documents of each class from %s to %s" % (N, in_dir, out_dir)
	else:
		print "Counting documents in ", in_dir
		
	main(in_dir, N, out_dir)
	
