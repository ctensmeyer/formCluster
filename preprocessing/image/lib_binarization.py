
import math

def build_histogram(histogram, iterable):
	_histo = histogram.copy()
	for item in iterable:
		_histo[item] = _histo[item] + 1 if item in _histo else 1
	return _histo


def normalize_histogram(histogram):
	_sum = float(sum(histogram.values()))
	return {key: val / _sum for key, val in histogram.iteritems()}


_2rootpi = math.sqrt(2 * math.pi)
def gauss(x, sig, mu=0):
	const = 1.0 / (sig * (_2rootpi))
	return const * math.exp( -( (x - mu) ** 2) / (2. * sig * sig ))


def otsu(distro, th_max):
	'''
	Threshold means that pixels less than or equal to threshold should be 0
	TODO: optimize taking advantage of incrementally changing cluster assignments
	'''
	
	otsu_thresh = -1
	max_result = -1

	max_val = max(distro.keys())
	th_max = min(th_max, max_val)
	for th in xrange(1, th_max + 1):
		num_c2 = 0
		sum_c2 = 0

		num_c1 = 0
		sum_c1 = 0
		# c1
		for val in xrange(0, th + 1):
			if val in distro:
				num_c1 += distro[val]
				sum_c1 += val * distro[val]
				
		for val in xrange(th + 1, max_val):
			if val in distro:
				num_c2 += distro[val]
				sum_c2 += val * distro[val]
					
		mean_c1 = sum_c1 / num_c1 if num_c1 else 0
		mean_c2 = sum_c2 / num_c2 if num_c2 else 0
		diff = mean_c2 - mean_c1
		diff_sq = diff ** 2

		result = diff_sq * num_c1 * num_c2
		result *= gauss(th, 150, 127)
		# print "Threshold: %d Result: %f" % (th, result)
		if result > max_result:
			max_result = result
			otsu_thresh = th

	return otsu_thresh

if __name__ == "__main__":
	for x in xrange(255):
		print x, gauss(x, 100, 127)

