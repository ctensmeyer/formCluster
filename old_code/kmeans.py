
import random
import math

class KMeans:

	def __init__(self, data, K):
		'''
		data must be a list of iterables (tuples or lists)
		'''
		self.K = K
		self.data = data
		self.N = len(data[0])
		self.M = len(data)

	def display(self):
		print "Data size: ", self.M
		for k in xrange(self.K):
			if not self.included[k]:
				continue
			print "Cluster %d: size: %d mean: %s std_dev: %s" % (k, self.counts[k], self.means[k], self.std_devs[k])

	def largest_cluster(self):
		k = self.counts.index(max(self.counts))
		return self.means[k]


	def ranges(self):
		r = list()
		for dim in xrange(self.N):
			domain = [datum[dim] for datum in self.data]
			r.append( (min(domain), max(domain)) )
		return r


	def init_model(self):
		means = []
		r = self.ranges()
		for k in xrange(self.K):
			mean = []
			for dim in xrange(self.N):
				val = random.random() * (r[dim][1] - r[dim][0]) + r[dim][0]
				mean.append(val)
			means.append(mean)
		self.means = means
		self.assignments = [random.randint(0, self.K-1) for x in xrange(self.M)]
		self.std_devs = [0 for k in xrange(self.K)]
		self.included = [True for k in xrange(self.K)]


	# return cluster idx of nearest cluster mean to datum
	def assign(self, datum):
		_min = 2 ** 31
		_min_idx = -1
		for k in xrange(self.K):
			if not self.included[k]:
				continue
			d = self.dist(datum, self.means[k])
			if d < _min:
				_min = d
				_min_idx = k
		return _min_idx


	def assign_all(self):
		for m in xrange(self.M):
			datum = self.data[m]
			new_assignment = self.assign(datum)
			if new_assignment != self.assignments[m]:
				self.keepGoing = True  # we have not yet reached equalibrium
				self.assignments[m] = new_assignment

	def compute_means(self):
		self.counts = [0 for k in xrange(self.K)]
		sums = [ [0.0 for n in xrange(self.N)] for k in xrange(self.K) ]

		# get cluster membership counts and by cluster sum of all data values
		for m in xrange(self.M):
			datum = self.data[m]
			assignment = self.assignments[m]
			self.counts[assignment] += 1
			_sum = sums[assignment]
			for n in xrange(self.N):
				_sum[n] += datum[n]
		#print "Counts: ", self.counts
		#print "Sums: ", sums

		# normalize and update means
		for k in xrange(self.K):
			count = self.counts[k]
			if count == 0:
				self.included[k] = False

			if not self.included[k]:
				continue

			new_mean = sums[k]
			for n in xrange(self.N):
				# what to do if count is 0?
				new_mean[n] /= count 
			self.means[k] = new_mean
		#print "Means: ", self.means

		# calculate standard deviations
		self.std_devs = [0.0 for k in xrange(self.K) ]
		for m in xrange(self.M):
			datum = self.data[m]
			assignment = self.assignments[m]
			self.std_devs[assignment] += self.dist(datum, self.means[assignment]) ** 2

		for k in xrange(self.K):
			if not self.included[k]:
				continue
			count = self.counts[k]
			self.std_devs[k] /= float(count)
			self.std_devs[k] = math.sqrt(self.std_devs[k])
		#print "Std_devs: ", self.std_devs


	def printAll(self):
		self.display()
		print
		for m in xrange(self.M):
			print self.data[m], " assigned to:", self.assignments[m]
			

	def cluster(self):
		self.init_model()
		self.keepGoing = True
		while self.keepGoing:
			self.keepGoing = False
			self.assign_all()
			self.compute_means()


	def dist(self, d1, d2):
		val = 0.0
		for dim in xrange(self.N):
			val += (d1[dim] - d2[dim]) ** 2
		return math.sqrt(val)

