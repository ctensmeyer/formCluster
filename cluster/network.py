
import utils

# this version turns all cells of a each mat into composites, which are then combined
class LinearNetwork:
	
	def __init__(self, num_feature_types, num_rows, num_cols, lr=0.1):
		self.num_feature_types = num_feature_types
		self.num_rows = num_rows
		self.num_cols = num_cols
		self.lr = lr
		self.region_weight_mats = [ self._uniform_mat()  for f in xrange(num_feature_types) ]
		self.region_combining_weights = self._uniform_list(self.num_feature_types)
		self.global_weights = self._uniform_list(self.num_feature_types)
		self.final_global_weight = 0.5
		self.final_region_weight = 0.5

	def display(self):
		print "final weights:\n\tglobal: %.3f\n\tregion: %.3f\n" % (self.final_global_weight, self.final_region_weight)
		for x in xrange(self.num_feature_types):
			print "Feature", x
			print "\tglobal: %.3f" % self.global_weights[x]
			print "\tregion: %.3f" % self.region_combining_weights[x]
			print "\tregion weight mat:"
			utils.print_mat(utils.apply_mat(self.region_weight_mats[x], lambda x: "%.3f" % x))
			print

	def _uniform_mat(self):
		div = float(self.num_rows * self.num_cols)
		return [ [1 / div] * self.num_cols for x in xrange(self.num_rows)]

	def _uniform_list(self, n):
		return [ 1.0 / n for x in xrange(n) ]

	def similarity(self, global_vals, region_vals):
		self.region_values = self.calc_region_values(region_vals)
		self.region_values_combined = sum(map(lambda x, y: x * y, self.region_values, self.region_combining_weights))
		self.global_values_combined = sum(map(lambda x, y: x * y, global_vals, self.global_weights))
		return self.region_values_combined * self.final_region_weight + self.global_values_combined * self.final_global_weight

	def backprop(self, global_vals, region_vals, target):
		sim = self.similarity(global_vals, region_vals)

		# calculate errs
		final_err = target - sim
		region_combined_err = final_err * self.final_region_weight
		global_combined_err = final_err * self.final_global_weight
		region_errs = map(lambda x: x * region_combined_err, self.region_combining_weights)

		# update weights
		self.final_global_weight += self.lr * final_err * self.global_values_combined
		if self.final_global_weight < 0:
			self.final_global_weight = 0
		self.final_region_weight += self.lr * final_err * self.region_values_combined
		if self.final_region_weight < 0:
			self.final_region_weight = 0
		for x in xrange(len(self.global_weights)):
			self.global_weights[x] += self.lr * global_combined_err * global_vals[x]
			if self.global_weights[x] < 0:
				self.global_weights[x] = 0
			self.region_combining_weights[x] += self.lr * region_combined_err * self.region_values[x]
			if self.region_combining_weights[x] < 0:
				self.region_combining_weights[x] = 0
			for r in xrange(self.num_rows):
				for c in xrange(self.num_cols):
					self.region_weight_mats[x][r][c] += self.lr * region_errs[x] * region_vals[x][r][c]
					if self.region_weight_mats[x][r][c] < 0:
						self.region_weight_mats[x][r][c] = 0

		# norm weights at each layer
		s = self.final_global_weight + self.final_region_weight
		self.final_global_weight /= s
		self.final_region_weight /= s

		self.global_weights = utils.norm_list(self.global_weights)
		self.region_combining_weights = utils.norm_list(self.region_combining_weights)
		for x in xrange(len(self.region_weight_mats)):
			self.region_weight_mats[x] = utils.norm_mat(self.region_weight_mats[x])

	def calc_region_values(self, mats):
		mult_mats = map(lambda mat, weights: utils.mult_mats([mat, weights]), mats, self.region_weight_mats)
		return map(lambda mat: sum(map(sum, mat)), mult_mats)


class WeightedAverageNetwork:
	
	def __init__(self, num_inputs, weights=None, default_lr=0.1,  empty_val=None, auto_norm=True):
		self.num_inputs = num_inputs
		self.lr = default_lr
		self.weights = self._uniform_weights() if weights is None else weights
		assert len(self.weights) == self.num_inputs
		self.empty_val = empty_val
		self.auto_norm = auto_norm
		if self.auto_norm:
			self.norm_weights()

	def _uniform_weights(self):
		return utils.norm_list([1] * self.num_inputs)

	def weight_sum(self):
		return sum(self.weights)

	def wavg(self, inputs):
		s = 0.0
		ws = 0.0
		for i in xrange(self.num_inputs):
			if inputs[i] != self.empty_val:
				s += inputs[i] * self.weights[i]
				ws += self.weights[i]
		wavg = s / ws if ws else 0.0
		return wavg

	def learn(self, inputs, target, lr=None):
		'''
		The update rule is similar to a simple perceptron update rule, but because the output
			is divided by the L1 of the weights, it is different.  Essentially, the mangitude of
			each weight change is proportional to the difference between the output and the weight's
			input.  Also as weights get big, the updates get smaller.
		'''
		if lr is None:
			lr = self.lr
		s = 0.0
		ws = 0.0
		for i in xrange(self.num_inputs):
			if inputs[i] != self.empty_val:
				s += inputs[i] * self.weights[i]
				ws += self.weights[i]
		wavg = s / ws if ws else 0.0
		err = target - wavg
		for i in xrange(self.num_inputs):
			if inputs[i] != self.empty_val:
				self.weights[i] += self.lr * err * (inputs[i] - wavg) / ws
				if self.weights[i] < 0:
					self.weights[i] = 0
		if self.auto_norm:
			self.norm_weights()

	def norm_weights(self):
		self.weights = utils.norm_list(self.weights)

def testlinear():
	network = LinearNetwork(1, 2, 2)
	global_vals = [1]
	region_vals = [[ [1, .1], [.2, .3] ]]

	print "Initial"
	network.display()
	for x in xrange(100):
		print "\nIteration", x, "\tTarget 1"
		print network.similarity(global_vals, region_vals)
		network.backprop(global_vals, region_vals, 1)
		network.display()

	for x in xrange(100):
		print "\nIteration", x, "\tTarget 0"
		print network.similarity(global_vals, region_vals)
		network.backprop(global_vals, region_vals, 0)
		network.display()

def testwavg():
	network = WeightedAverageNetwork(3, default_lr=.3)
	datum = (.8, .5, .2)

	print "Weighted Average Network"
	for x in xrange(10):
		print "\nIteration", x, "\tTarget 1"
		print network.wavg(datum)
		network.learn(datum, 1)
		print network.weights, network.weight_sum()

	for x in xrange(10):
		print "\nIteration", x, "\tTarget 0"
		print network.wavg(datum)
		network.learn(datum, 0)
		print network.weights, network.weight_sum()

		
if __name__ == "__main__":
	#testlinear()
	testwavg()

		
