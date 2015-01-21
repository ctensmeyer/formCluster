
import os
import sys
import cv2

def extract_surf(im_file, out_file, limit):
	im = cv2.imread(im_file, 0)
	surf = cv2.SURF(1000)
	surf.upright = True
	surf.extended = True
	kp, des = surf.detectAndCompute(im, None)

	#im2 = cv2.drawKeypoints(im, kp, None, (255, 0, 0), 4)
	#cv2.imwrite(sys.argv[2], im2)

	out = open(out_file, 'w')
	for x, tup in enumerate(zip(kp, des)):
		k, d = tup
		out.write("%s\n" % (' '.join(map(str, list(k.pt) + list(d)))))
		if x >= limit:
			break
	out.close()
	
if __name__ == "__main__":
	indir = sys.argv[1]
	outdir = sys.argv[2]
	limit = int(sys.argv[3])
	count = 0
	for subdir in os.listdir(indir):
		rdir = os.path.join(indir, subdir)
		todir = os.path.join(outdir, subdir)
		try:
			os.makedirs(todir)
		except:
			pass
		for f in os.listdir(rdir):
			if f.endswith(".jpg"):
				print f
				im_file = os.path.join(rdir, f)
				out_file = os.path.join(todir, f.replace(".jpg", "_surf.txt"))
				extract_surf(im_file, out_file, limit)
				count += 1
				if count % 100 == 0:
					print "Processed %d Documents" % count

