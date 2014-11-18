
import os
import sys
import cv2
import Image
import string
import image.line_extract_lib as line_lib
import ocr.ocr as ocr
import kmedoids
import random
import numpy as np
import scipy.spatial.distance
import multiprocessing

# Parameters
_surf_upright = True
_surf_extended = False
_surf_threshold = 10000
_surf_threshold_low = 3000
_num_surf_features = 10000
_min_surf_features = 5000

_codebook_size = 300
_perc_docs_for_codebook = 0.05
_max_surf_features = _codebook_size * 100
_max_k_medoids_iters = 30

_surf_instance = cv2.SURF(_surf_threshold)
_surf_instance.upright = _surf_upright
_surf_instance.extended = _surf_extended

_surf_instance_low = cv2.SURF(_surf_threshold_low)
_surf_instance_low.upright = _surf_upright
_surf_instance_low.extended = _surf_extended

_cache_codebook_file = "./codebook.pckl"
_read_cache = False
_write_cache = True

def get_id(img_file):
	return os.path.splitext(img_file)[0]


def get_size(img_file):
	im = Image.open(img_file)
	size = im.size
	del im
	return size


def get_hv_lines(h_line_file, v_line_file):
	h_img = Image.open(h_line_file)
	h_ccs = line_lib.get_line_ccs(h_img)
	h_lines = map(line_lib.extract_line_from_cc, h_ccs)

	v_img = Image.open(v_line_file)
	v_ccs = line_lib.get_line_ccs(v_img)
	v_lines = map(line_lib.extract_line_from_cc, v_ccs)

	return h_lines, v_lines


def create_verify_file(h_lines, v_lines, line_verify_file, size):
	im = Image.new("RGB", size, "white")
	line_lib.draw_lines(h_lines, v_lines, im)
	im.save(line_verify_file)
	del im


def strip_prefix(s, prefix):
	if s.startswith(prefix):
		s = s[len(prefix):]
	return s


def get_label(label_file):
	contents = filter(lambda x: x in string.printable, open(label_file).read())
	contents = strip_prefix(contents, "UK1911Census_EnglandWales_")
	if contents == "Household100Names_08_01":
		contents = "Household40Names_07_01"
	return contents


def get_text_lines(ocr_file):
	return ocr.extract_text_lines(ocr_file)

def get_surfs(im_file, codebook):
	surfs = list()
	pts, ds = extract_surf_features(im_file)
	closest_code = lambda feature: scipy.spatial.distance.cdist(codebook, [feature], metric='cityblock').argmin()

	# populate the most fine grained partitions
	for pt, des in zip(pts, ds):
		idx = closest_code(des)
		surfs.append( (int(pt[0]), int(pt[1]), idx) )

	return surfs

def transfer2(args):
	transfer(*args)

def transfer(img_file, h_line_file, v_line_file, label_file, ocr_file, out_file, line_verify_file, codebook):
	print out_file
	_id = get_id(img_file)
	size = get_size(img_file)	
	label = get_label(label_file)
	h_lines, v_lines = get_hv_lines(h_line_file, v_line_file)
	create_verify_file(h_lines, v_lines, line_verify_file, size)
	text_lines = get_text_lines(ocr_file)
	surfs = get_surfs(img_file, codebook)
	f = open(out_file, 'w')
	f.write("%s\n" % _id)
	f.write("%s\n" % label)
	f.write("%d %d\n\n" % (size[0], size[1]))

	for line in text_lines:
		f.write("%d %d %d %d %s\n" % (line[1][0], line[1][1], line[2][0], line[2][1], line[0]) )

	f.write("\n")
	for line in h_lines:
		f.write("%d %d %d %d\n" % (line[4], line[3], line[1], line[2]))

	f.write("\n")
	for line in v_lines:
		f.write("%d %d %d %d\n" % (line[3], line[4], line[1], line[2]))

	f.write("\n")
	f.write("%d\n" % _codebook_size)
	for line in surfs:
		f.write("%d %d %d\n" % line)

	f.write("\n")


def extract_surf_features(im_file):
	im = cv2.imread(im_file, 0)
	kps, deses = _surf_instance.detectAndCompute(im, None)
	if len(kps) < _min_surf_features:
		kps, deses = _surf_instance_low.detectAndCompute(im, None)
	pts = np.array(map(lambda kp: kp.pt, kps[:_num_surf_features]))
	ds = deses[:_num_surf_features] + 0
	del deses
	return (pts, ds)
	

def create_codebook(indir):
	if _read_cache and os.path.exists(_cache_codebook_file):
		print "\tReading Cache"
		f = open(_cache_codebook_file)
		try:
			codebook = cPickle.load(f)
			f.close()
			print "Done\n"
			return codebook
		except Exception as e:
			print "\tError loading codebook:", e
			print "\tComputing From scratch"
			f.close()

	# sample some files
	im_files = list()
	for subdir in os.listdir(indir):
		rdir = os.path.join(indir, subdir)
		for f in os.listdir(rdir):
			if f.endswith(".jpg"):
				img_file = os.path.join(rdir, f)
				im_files.append(img_file)
	codebook_files = random.sample(im_files, int(_perc_docs_for_codebook * len(im_files)))

	# construct the codebook
	surfs = np.concatenate(map(lambda x: extract_surf_features(x)[1], codebook_files))
	np.random.shuffle(surfs)
	surfs = surfs[:_max_surf_features]
	distances = scipy.spatial.distance.pdist(surfs, 'cityblock')
	distances = scipy.spatial.distance.squareform(distances)
	indices = kmedoids.cluster(distances, k=_codebook_size, maxIters=_max_k_medoids_iters)[1]
	codebook = surfs[indices]

	if _write_cache:
		print "\tWriting Codebook to Cache"
		f = open(_cache_codebook_file, 'w')
		try:
			cPickle.dump(codebook, f)
		except Exception as e:
			print "\tCould not write to cache:", e
		f.close()
	print "Done\n"

	return codebook

if __name__ == "__main__":
	indir = sys.argv[1]
	outdir = sys.argv[2]
	codebook = create_codebook(indir)
	args = list()
	for subdir in os.listdir(indir):
		rdir = os.path.join(indir, subdir)
		todir = os.path.join(outdir, subdir)
		try:
			os.makedirs(todir)
		except:
			pass
		for f in os.listdir(rdir):
			if f.endswith(".jpg"):
				img_file = os.path.join(rdir, f)
				h_line_file = os.path.join(rdir, f.replace(".jpg", "_linesH.pgm"))
				v_line_file = os.path.join(rdir, f.replace(".jpg", "_linesV.pgm"))
				label_file = os.path.join(rdir, f.replace(".jpg", "_FormType.txt"))
				ocr_file = os.path.join(rdir, f.replace(".jpg", ".xml"))
				out_file = os.path.join(todir, f.replace(".jpg", ".txt"))
				verify_file = os.path.join(todir, f.replace(".jpg", "_verify.png"))
				if not os.path.exists(out_file):
					#print out_file
					#transfer(img_file, h_line_file, v_line_file, label_file, ocr_file, out_file, verify_file, codebook)
					args.append( (img_file, h_line_file, v_line_file, label_file, ocr_file, out_file, verify_file, codebook) )
	pool = multiprocessing.Pool(4)
	pool.map(transfer2, args, 10)
	
	
