
import math
from imageutils import gauss


def avg(profile):
	'''
	:return: float average value of profile
	'''
	return reduce(lambda x, y: x + y, profile) / float(len(profile))


def stdev(profile):
	'''
	:return: float standard deviation of profile
	'''
	a = avg(profile)
	return math.sqrt(reduce(lambda x, y: x + (y - a) ** 2, profile) / float(len(profile)))

def _1d_gauss_table(window, sigma):
	table = []
	for x in xrange(window / 2 + 1):
		table.append(gauss(x, sigma))
	return table
	

def blur_gauss(profile, window, spacial_sigma):
	'''
	Perform a gaussian bluring kernel convolution on the profile
	:param profile: seq of num
	:window: odd-int window size for blurring kernel
	:spacial_sigma: num param to the gaussian function
	:return: new blurred list of num of same length as profile
	'''
	result = []
	l = len(profile)
	_table = _1d_gauss_table(window + 1, spacial_sigma)
	for x in xrange(l):
		weight_sum = 0.0
		pixel_sum = 0.0
		for offset in xrange(-1 * window / 2, (window + 1) / 2):
			idx = x + offset
			if idx >= 0 and idx < l:
				weight = _table[abs(offset)]
				weight_sum += weight
				pixel_sum += weight * profile[idx]
		new_value = pixel_sum / weight_sum
		result.append(new_value)
	assert len(result) == l
	return result


def blur_bilateral(profile, window, spacial_sigma, value_sigma):
	'''
	Perform a gaussian bluring kernel convolution on the profile
	:param profile: seq of num
	:window: odd-int window size for blurring kernel
	:spacial_sigma: num param to the gaussian function
	:value_sigma: num param to the gaussian function
	:return: new blurred list of num of same length as profile
	'''
	result = []
	l = len(profile)
	#_spacial_table = _1d_gauss_table(window + 1, spacial_sigma)
	#_value_table = _1d_gauss_table(int(max(profile) - min(profile)) + 1, value_sigma)
	for x in xrange(l):
		weight_sum = 0.0
		pixel_sum = 0.0
		for offset in xrange(-1 * window / 2, (window + 1) / 2):
			idx = x + offset
			if idx >= 0 and idx < l:
				#spacial_weight = _spacial_table[abs(offset)]
				spacial_weight = gauss(abs(offset), spacial_sigma)

				# we approximate here in order to use the look-up table
				val_diff = int(round(profile[x] - profile[idx]))
				#value_weight = _value_table[abs(val_diff)]
				value_weight = gauss(abs(val_diff), value_sigma)

				weight = spacial_weight * value_weight
				weight_sum += weight
				pixel_sum += weight * profile[idx]
		new_value = pixel_sum / weight_sum
		result.append(new_value)
	assert len(result) == l
	return result


def blur_uniform(profile, window):
	'''
	Perform a uniform bluring kernel convolution on the profile
	:param profile: seq of num
	:window: odd-int window size for blurring kernel
	:return: new blurred list of num of same length as profile
	'''
	result = []
	l = len(profile)
	for x in xrange(l):
		count = 0
		s = 0.0
		for offset in xrange(-1 * window / 2, (window + 1) / 2):
			idx = x + offset
			if idx >= 0 and idx < l:
				count += 1
				s += profile[idx]
		result.append( s / count)
	assert len(result) == l
	return result


def scale(profile, factor):
	'''
	:param profile: seq of num
	:param factor: float multiplier for each element of profile
	'''
	return [x * factor for x in profile]


def derivative(profile):
	'''
	:param profile: seq of num
	:return: new derivative list 
	'''
	result = [0]
	for x in xrange(1, len(profile) - 1):
		n = (-1 * profile[x-1] + profile[x+1]) / 2
		result.append(n)
	result.append(0)
	return result

# This returns a list of indices of the extrema.
# Remember endpoints are extrema too
def extract_extrema(profile):
	'''
	:param profile: seq of num
	:return: list of idx of profile where extrema lie
	'''
	result = []
	derv = derivative(profile)
	prev = derv[0]
	for x in xrange(1, len(derv)):
		val = derv[x]
		if (prev <= 0) != (val <= 0):
			result.append(x)
		prev = val
	return result
	

def filter_extrema(profile, extrema, min_support, min_delta):
	'''
	:param profile: seq of num
	:param extrema: seq of idx of the extrema of profile
	:param min_support: num minimum support for the filtered extrema
	:param min_delta: num minimum absolute difference between peak and valley
	'''
	filtered = []
	prev_prof = 0
	for i_prof in extrema:
		# calculate support, delta for each extrema
		support = i_prof - prev_prof
		delta = abs(profile[i_prof] - profile[prev_prof])

		if support >= min_support and delta >= min_delta:
			filtered.append(i_prof)
		prev_prof = i_prof
	return filtered


def extract_support(extrema):
	'''
	:return: list of support of the extrema
	'''
	result = [0]
	for x in xrange(1, len(extrema)):
		result.append(extrema[x] - extrema[x-1])
	return result
		

def extract_delta(profile, extrema):
	'''
	:return: list of signed deltas
	'''
	deltas = []
	prev_prof = 0
	for i_prof in extrema:
		# calculate support, delta for each extrema
		delta = profile[i_prof] - profile[prev_prof]
		deltas.append(delta)
		prev_prof = i_prof
	return deltas
	

def extrema_as_points(extrema, profile):
	'''
	:return list of (x, y) coordinates for the extrema points
	'''
	points = []
	for x in extrema:
		points.append( (x, profile[x]) )
	return points


def consolodate(*lists):
	'''
	:param lists: list of lists
	:return: list of tuple of elements in the same indexed position.
		len of result is len of shortest input
	'''
	result = []
	for x in xrange(min([len(l) for l in lists])):
		result.append(tuple([l[x] for l in lists]))
	return result


def _extrema_matcher_creator(_ex_1, pr_1, _ex_2, pr_2, x_allowance, mismatch_penalty):
	'''
	Create matcher function to be used in _edit_distance
	'''
	delta_avg1 = avg(map(lambda x: abs(x), extract_delta(pr_1, _ex_1)))
	delta_avg2 = avg(map(lambda x: abs(x), extract_delta(pr_2, _ex_2)))
	
	def extrema_matcher(i, j, ex_1, ex_2):
		#print ex_1[i], ex_2[j]
		support1 = (ex_1[i] - ex_1[i-1] if i else ex_1[i]) / float(len(pr_1))
		support2 = (ex_2[j] - ex_2[j-1] if j else ex_2[j]) / float(len(pr_2))
		sup_ratio = max(support1 / support2, support2 / support1)
		#print '\tsup:', support1, support2, sup_ratio

		# handle peak compared to valley
		delta1 = (pr_1[ex_1[i]] - pr_1[ex_1[i-1]] if i else pr_1[ex_1[i]]) / delta_avg1
		delta2 = (pr_2[ex_2[j]] - pr_2[ex_2[j-1]] if j else pr_2[ex_2[j]]) / delta_avg2
		del_ratio = max(abs(delta1 / delta2), abs(delta2 / delta1))
		#print '\tdelta:', delta1, delta2, del_ratio
		delta_penalty = mismatch_penalty if (delta1 >= 0) != (delta2 >= 0) else 0

		x1 = ex_1[i] / float(len(pr_1))
		x2 = ex_2[j] / float(len(pr_2))
		x_del = abs(x1 - x2)
		x_penalty = max(0, (x_del - x_allowance) * 10)
		#print '\tpos:', x1, x2, x_del

		dist = math.log(sup_ratio, 2) + math.log(del_ratio, 2) + delta_penalty + x_penalty
		
		return dist

	return extrema_matcher
	

def extrema_distance(pr_1, ex_1, pr_2, ex_2):
	'''
	Top-Level distance metric
	'''
	match_f = _extrema_matcher_creator(ex_1, pr_1, ex_2, pr_2, 0.02, 3)
	indel_cost = 2
	return _edit_distance(ex_1, ex_2, indel_cost, match_f)


# Operations include skip or match
# Match cost is dependent on the two extrema
def _edit_distance(s, t, id_cost, match_f):
	'''
	:param s: 0 sequence 1
	:param t: 0 sequence 2
	:id_cost: num Cost of an Insertion or Deletion operation
	:match_f: func (idx1, idx2, s, t) -> num  Cost of matching
	:return: Edit distance between s and t
	'''
	l1 = len(s) + 1 # width
	l2 = len(t) + 1 # height
	d = [ [x * id_cost for x in xrange(l2)] ]

	for i in xrange(1, l1):
		d.append([i * id_cost])
		for j in xrange(1, l2):
			_del = d[i-1][j] + id_cost
			_ins = d[i][j-1] + id_cost
			_match = match_f(i-1, j-1, s, t) + d[i-1][j-1]
			d[i].append(min(_del, _ins, _match))
	i = l1 - 1
	j = l2 - 1
	trace = []
	while (i > 0 and j > 0):
		val = d[i][j]
		_del = d[i-1][j] + id_cost
		_ins = d[i][j-1] + id_cost
		_match = match_f(i-1, j-1, s, t) + d[i-1][j-1]
		_min = min(_del, _ins, _match)
		if _min == _del:
			i -= 1
		elif _min == _ins:
			j -= 1
		elif _min == _match:
			trace.append( ( i - 1, j - 1, _match - d[i-1][j-1]) )
			i -= 1
			j -= 1
	trace.reverse()
	final_val =  d[l1 - 1][l2 - 1] 
	norm = final_val / (len(s) + len(t))
	return (norm, trace)
		
	
if __name__ == "__main__":
	print _edit_distance("abcdef", "abc", 1,  lambda i, j, s, t: -2 if s[i] == t[j] else 1)

