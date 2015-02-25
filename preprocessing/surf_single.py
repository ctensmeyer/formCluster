
import os
import sys
import cv2

def extract_surf(im_file, out_file, limit):
	im = cv2.imread(im_file, 0)
	surf = cv2.SURF(limit)
	surf.upright = True
	surf.extended = True
	kp, des = surf.detectAndCompute(im, None)

	im2 = cv2.drawKeypoints(im, kp, None, (255, 0, 0), 4)
	cv2.imwrite(out_file, im2)
	
	print len(kp)

	exit()

	out = open(out_file, 'w')
	for x, tup in enumerate(zip(kp, des)):
		k, d = tup
		out.write("%s\n" % (' '.join(map(str, list(k.pt) + list(d)))))
		if x >= limit:
			break
	out.close()
	
if __name__ == "__main__":
	im_file = sys.argv[1]
	out_file = sys.argv[2]
	limit = int(sys.argv[3])
	extract_surf(im_file, out_file, limit)

