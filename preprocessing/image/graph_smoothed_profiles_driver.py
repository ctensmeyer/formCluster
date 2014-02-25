
import os
import sys
import imageutils
import datautils
import signalutils
import Image

IMAGE_EXT = '.png'

def f(sp_sigma_divisor, value_sigma, num_times):
	def _preprocess(profile):
		#spacial_sigma = 1
		#value_sigma = 100000
		window = len(profile) / 100
		spacial_sigma = len(profile) / sp_sigma_divisor
		#profile = signalutils.blur_uniform(profile, 5)
		#profile = signalutils.blur_gauss(profile, window, spacial_sigma)
		#profile = signalutils.blur_bilateral(profile, window, spacial_sigma, value_sigma)
		for x in xrange(num_times):
			profile = signalutils.blur_bilateral(profile, window, spacial_sigma, value_sigma)
		return profile
	return _preprocess


def main(in_dir, image_info_dir, out_dir):
	#sigmas = [5, 10, 20, 30] 
	for num_times in [2]:
		for value_sigma in [8]:
			for sp_sigma_divisor in [250.0]:
				for filename in os.listdir(in_dir):
					if not filename.endswith(IMAGE_EXT):
						continue
					in_image = os.path.join(in_dir, filename)
					basename = os.path.splitext(filename)[0]
					image_info = os.path.join(image_info_dir, basename + '.xml')
					filename = "%s_%d_v%d_s%d.jpg" % (basename, num_times, value_sigma, sp_sigma_divisor)
					out_image = os.path.join(out_dir, filename)

					#profiles = datautils.get_profiles(image_info, preprocess=datautils._preprocess,
					#											  normalize=True)
					profiles = datautils.get_profiles(image_info,
						preprocess=f(sp_sigma_divisor, value_sigma, num_times), normalize=True)
					#profiles = datautils.get_profiles(image_info, preprocess=None, normalize=True)

					# draw the vertical profile
					vert = Image.open(in_image)
					profile = signalutils.scale(profiles['Vertical'], 10)
					vert = imageutils.graph_profile(vert, profile, color='green', width=5)

					# draw vertical extrema
					extrema = signalutils.extract_extrema(profile)
					points = signalutils.extrema_as_points(extrema, profile)
					vert = imageutils.graph_points(vert, points, color='purple', width=7)

					# draw filtered vertical extream
					min_support = 0
					#deltas = map(lambda x: abs(x), signalutils.extract_delta(profile, extrema))
					#min_delta = signalutils.avg(deltas) - 0.5 * signalutils.stdev(deltas)
					min_delta = 5
					filtered = signalutils.filter_extrema(profile, extrema, min_support, min_delta)
					points = signalutils.extrema_as_points(filtered, profile)
					vert = imageutils.graph_points(vert, points, color='black', width=7)

					# draw the horizontal profile
					horz = vert
					horz = horz.rotate(90)
					profile = signalutils.scale(profiles['Horizontal'], 10)
					horz = imageutils.graph_profile(horz, profile, color='blue', width=5)

					# draw horizontal extrema
					extrema = signalutils.extract_extrema(profile)
					points = signalutils.extrema_as_points(extrema, profile)
					horz = imageutils.graph_points(horz, points, color='orange', width=7)

					# draw filtered horizontal extream
					min_support = 0
					#deltas = map(lambda x: abs(x), signalutils.extract_delta(profile, extrema))
					#min_delta = signalutils.avg(deltas) - 0.5 * signalutils.stdev(deltas)
					min_delta = 5
					filtered = signalutils.filter_extrema(profile, extrema, min_support, min_delta)
					points = signalutils.extrema_as_points(filtered, profile)
					horz = imageutils.graph_points(horz, points, color='black', width=7)
					horz = horz.rotate(270)

					horz.save(out_image)



if __name__ == "__main__":
	if len(sys.argv) < 4:
		raise Exception("[in_dir image_info_dir out_dir]")
	in_dir = sys.argv[1]
	image_info_dir = sys.argv[2]
	out_dir = sys.argv[3]
	main(in_dir, image_info_dir, out_dir)

	
