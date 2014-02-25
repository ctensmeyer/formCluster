
# perform two passes of a 1D bilateral filter on the projection profiles
def bilateral(prof):
	window = len(prof) / 200
	value_sigma = 20 
	spacial_sigma = len(prof) / 200.0
	prof = signalutils.blur_bilateral(prof, window, spacial_sigma, value_sigma)
	prof = signalutils.blur_bilateral(prof, window, spacial_sigma, value_sigma)
	return prof

def main():
	# Indicate that we want bilateral smoothing.  Original results used uniform smoothing
	processors = {'all': bilateral}

	# Calculate the projection profiles on all images in a given directory
	profiles = calc_all_profiles(sys.argv[1], processors, normalize=True)

	# Perform the pairwise comparisons between images using projection profiles
	sim_mat = form_sim_mat(profiles)

	# Write similarity matrix out to a file
	write_data(sys.argv[2], sim_mat, sorted(profiles.keys()))

def form_sim_mat(all_profiles):
	'''
	:param all_profiles: dict: {identifier : {'Horizontal' : <profile>, 'Vertical' : <profile>, 'dimensions' : (width, height)}, }
	:return: list of lists of similarity scores (float).  Row/Col ordering is sorted order of identifiers
	'''
	sim_mat = []
	for id_1 in sorted(all_profiles.keys()):
		row = []
		for id_2 in sorted(all_profiles.keys()):
			# iterate over all pairs of images
			if id_1 == id_2:
				# skip comparison if the two images are equal
				row.append(0.0)
				continue
			_sum = 0
			# first compare Horizontal profiles, then Vertical profiles
			for _type in ['Horizontal', 'Vertical']:
				# pull out profile data
				pr_1 = all_profiles[id_1][_type]
				pr_2 = all_profiles[id_2][_type]

				# Find the extrema points of each profile
				ex_1 = signalutils.extract_extrema(pr_1)
				ex_2 = signalutils.extract_extrema(pr_2)

				# Filter out insignificant extrema
				min_support_1 = 0
				min_delta_1 = 3
				f_ex_1 = signalutils.filter_extrema(pr_1, ex_1, min_support_1, min_delta_1)

				min_support_2 = 0
				min_delta_2 = 3
				f_ex_2 = signalutils.filter_extrema(pr_2, ex_2, min_support_2, min_delta_2)

				# Calculate the distance between two profiles based on the extrema
				_sum += signalutils.extrema_distance(pr_1, f_ex_1, pr_2, f_ex_2)[0] ** 2
			row.append(math.sqrt(_sum))
		sim_mat.append(row)
	return sim_mat

def extrema_distance(pr_1, ex_1, pr_2, ex_2):
	'''
	Top-Level distance metric
	'''
	# this function determines the cost of matching two extrema
	# It does so based on x, y distances from the previous extrema and the absolute x position of the extrema
	match_f = _extrema_matcher_creator(ex_1, pr_1, ex_2, pr_2, 0.02, 3)

	# cost of throwing out extrema from the sequence
	indel_cost = 2

	# perform the edit distance based on the matching function and the indel cost
	# This is normalized to account for the length of both sequences of extrema
	return _edit_distance(ex_1, ex_2, indel_cost, match_f)



