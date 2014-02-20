
import xml.etree.ElementTree as ET
import os
import sys
import utils.signalutils as sigutils
import utils.datautils as datautils
from utils.graph_profiles import f as bilateral

sp = 20
val = 5
times = 4

def main(inputdir, outputdir):
	all_profiles = datautils.load_all_profiles(inputdir, normalize=True)
	all_profiles_smoothed = datautils.load_all_profiles(inputdir, preprocess=bilateral(sp, val, times), normalize=True)

	for filename in all_profiles:
		for _prof in ['Horizontal', 'Vertical']:
			basename = os.path.splitext(filename)[0] + "_" + _prof
			outfilename = os.path.join(outputdir, basename + '.plot')

			rough = all_profiles[filename][_prof]
			rough_ex = sigutils.extract_extrema(rough)
			rough_ex = sigutils.filter_extrema(rough, rough_ex, 0, 50)
			rough_deltas = sigutils.extract_delta(rough, rough_ex)
			rough_ex_plot = [0 for x in xrange(len(rough))]
			for x in xrange(len(rough_ex)):
				rough_ex_plot[rough_ex[x]] = rough_deltas[x]

			smooth = all_profiles_smoothed[filename][_prof]
			smooth_ex = sigutils.extract_extrema(smooth)
			smooth_ex = sigutils.filter_extrema(smooth, smooth_ex, 0, 50)
			smooth_deltas = sigutils.extract_delta(smooth, smooth_ex)
			smooth_ex_plot = [0 for x in xrange(len(smooth))]
			for x in xrange(len(smooth_ex)):
				smooth_ex_plot[smooth_ex[x]] = smooth_deltas[x]

			combined = sigutils.consolodate(xrange(len(rough)), rough, rough_ex_plot, smooth, smooth_ex_plot)
			datautils.write_cols(outfilename, combined)


if __name__ == "__main__":
	if len(sys.argv) < 2:
		raise Exception("[image_info_dir output_dir")
	inputdir = sys.argv[1] 
	outputdir = sys.argv[2]
	main(inputdir, outputdir)

