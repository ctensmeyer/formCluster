
import collections
import json
import utils
import math

_counts = ['TP', 'TN', 'FP', 'FN']
eps = 10e-10

class KnownClusterAnalyzer:

	def __init__(self, clusters, cluster_centers=None):
		self.clusters = clusters
		self.sort_by_size()

		self.create_cluster_ids()
		self.fill_in_cluster_labels()
		self.all_labels = self.get_all_labels()

		self.num_docs = sum(map(lambda cluster: len(cluster.members), clusters))
		self.label_pr_mats = self.calc_label_pr_mats()
		self.label_cluster_mat = self.calc_label_cluster_counts()
		self.total_counts = {count: 0 for count in _counts}
		for label in self.label_pr_mats:
			for count in _counts:
				n = self.label_pr_mats[label][count]
				self.total_counts[count] += n
		#self.print_assignments()

	def sort_by_size(self):
		self.clusters.sort(key=lambda cluster: len(cluster.members), reverse=True)

	def create_cluster_ids(self, prefix="cluster_"):
		num = 0
		for cluster in self.clusters:
			if cluster._id is None:
				#cluster._id = "%s%d" % (prefix, num)
				cluster._id = num
				num += 1

	def fill_in_cluster_labels(self):
		for cluster in self.clusters:
			if cluster.label is None:
				cluster.label = self.majority_label(cluster)

	def print_assignments(self):
		print "ALL DOCUMENT ASSIGNMENTS:"
		print "\tID\tMatch?\tTrue Label\tAssigned Label"
		print "\t-----------------------------------------------------------"
		for cluster, label in zip(self.clusters, self.cluster_labels):
			print
			for doc in cluster.members:
				print "\t" + "\t".join([str(doc._id), str(doc.label == label), doc.label, label])
		print
		print

	def get_all_labels(self):
		labels = set()
		for cluster in self.clusters:
			for doc in cluster.members:
				labels.add(doc.label)
		return labels

	def get_true_doc_histogram(self):
		counts = collections.defaultdict(int)
		for cluster in self.clusters:
			for doc in cluster.members:
				counts[doc.label] += 1
		return counts

	def get_doc_histogram(self):
		counts = collections.defaultdict(int)
		for cluster in self.clusters:
			counts[cluster.label] += len(cluster.members)
		return counts

	def get_cluster_histogram(self):
		histo = collections.Counter(map(lambda cluster: cluster.label, self.clusters))
		histo.update(self.all_labels)
		histo.subtract(self.all_labels)
		return histo
	
	def print_all(self):
		print
		self.print_general_info()
		self.print_histogram_info()
		self.print_cluster_sim_mat()
		self.print_label_conf_mat()
		self.print_label_cluster_mat()
		self.print_cluster_cohesion()
		self.print_label_info()
		self.print_metric_info()
		print

	def print_label_conf_mat(self):
		print "LABEL CONFUSION MATRIX:"
		print "\tRows are actual predictions.  Columns are true labels"
		mat = self.calc_conf_mat()
		labels = sorted(mat.keys())
		mat = utils.format_as_mat(mat)
		utils.insert_indices(mat)
		print "\tMat index\tLabel Name"
		for x,label in enumerate(labels):
			print "\t%d:\t%s" % (x, label)
		print
		utils.print_mat(mat)
		print
		print

	def print_label_cluster_mat(self):
		print "LABEL-CLUSTER MATRIX:"
		print "\tSeries of matricies.  Labels are rows.  Clusters are columns."
		mat = self.label_cluster_mat
		labels = sorted(mat.keys())
		mat = utils.format_as_mat(mat)
		clusters_per_mat = 20
		mats = utils.split_mat(mat, clusters_per_mat)  # 10 clusters per matrix

		print "\tMat index\tLabel Name"
		for x,label in enumerate(labels):
			print "\t%d:\t%s" % (x, label)
		print
		for x, mat in enumerate(mats):
			utils.insert_indices(mat, col_start=(clusters_per_mat * x))
			utils.print_mat(mat)
			print
		print 


	def print_cluster_sim_mat(self, min_size=10):
		print "CLUSTER SIM MATRIX:"
		clusters = filter(lambda cluster: len(cluster.members) > min_size, self.clusters)
		print "\tOnly showing the %d clusters containing more than %d elements" % (len(clusters), min_size)
		centers = map(lambda cluster: cluster.center, clusters)
		mat = utils.pairwise(centers, lambda doc1, doc2: doc1.dist(doc2))
		mat = utils.apply_mat(mat, lambda x: "%3.2f" % x)
		utils.insert_indices(mat)
		utils.print_mat(mat)
		print
		print

	def print_cluster_cohesion(self):
		print "CLUSTER COHESION:"
		print "\t\tAVG\tSTDDEV\tLEN"
		for x, cluster in enumerate(self.clusters):
			dists = map(lambda doc: doc.dist(cluster.center), cluster.members)
			avg = utils.avg(dists)
			sd = utils.stddev(dists)
			l = len(dists)
			print "\t%s:\t%3.2f\t%3.2f\t%d" % (x, avg, sd, l)
		print
		print

	def print_label_info(self):
		assigned_labels = set(map(lambda cluster: cluster.label, self.clusters))
		print "LABEL INFO:"
		print "\tThere are %d true labels" % len(self.all_labels)
		print "\tThere are %d assigned labels" % len(assigned_labels)
		print "\t%d labels were not assigned to any cluster" % (len(self.all_labels) - len(assigned_labels))
		print "\n\tAssigned Labels\n\t-------------------------"
		for label in sorted(assigned_labels):
			print "\t%s" % label
		print "\n\tTrue Labels not assigned\n\t-------------------------"
		for label in sorted(self.all_labels - assigned_labels):
			print "\t%s" % label
		print
		print

	def print_general_info(self):
		print "GENERAL INFO:"
		print "\tNumber of Documents: ", self.num_docs
		print "\tNumber of Classes: ", len(self.all_labels)
		print "\tNumber of Clusters: ", len(self.clusters)
		print
		print

	def print_histogram_info(self):
		print "HISTOGRAM INFO:"
		print "\tDocuments True Label\n\t-----------------------------"
		c = self.get_true_doc_histogram()
		for label in self.all_labels:
			print "\t%s:\t%d (%2.1f%%)" % (label, c[label], 100.0 * c[label] / self.num_docs)
		print "\n\tDocuments Assigned to Label\n\t-----------------------------"
		c = self.get_doc_histogram()
		for label in self.all_labels:
			print "\t%s:\t%d (%2.1f%%)" % (label, c[label], 100.0 * c[label] / self.num_docs)
		print "\n\tClusters Assigned to Label\n\t-----------------------------"
		c = self.get_cluster_histogram()
		for label in self.all_labels:
			print "\t%s:\t%d (%2.1f%%)" % (label, c[label], 100.0 * c[label] / len(self.clusters))
		print
		print

	def print_metric_info(self):
		print "METRIC INFO:"
		print "\tAccuracy: ", self.accuracy()
		print "\tV-measure: ", self.v_measure()
		print "\t\tHomogeneity: ", self.homogeneity()
		print "\t\tCompleteness: ", self.completeness()
		print "\tF1 Macro: ", self.F1_macro()
		print "\tF1 Micro: ", self.F1_micro()
		print "\tTotal PR/RC: ", self.PR()
		print "\tLabel break down:"
		for label in self.all_labels:
			print "\n\t\t%s:" % label
			print "\t\tF1: %.3f" % self.F1()
			print "\t\tPR: %.3f\tRC: %.3f" % (self.PR(label))
			s = "\t".join(map(lambda count: "%s: %d (%2.1f%%)" % (count, self.label_pr_mats[label][count],
						100.0 * self.label_pr_mats[label][count] / self.num_docs), _counts))
			print "\t\t%s" % s
		print
		print

	def calc_label_cluster_counts(self):
		'''
		:return: { label : { cluster_id : #occurances, }, }
		'''
		counts = {label: {cluster._id: 0 for cluster in self.clusters} for label in self.all_labels}
		for cluster in self.clusters:
			for doc in cluster.members:
				counts[doc.label][cluster._id] += 1
		return counts

	def calc_label_pr_mats(self):
		'''
		:return: {label : { count_type (TP, etc) : #occurances, }, }
		'''
		counts = {label: {count: 0 for count in ['TP', 'TN', 'FP', 'FN']} for label in self.all_labels}
		for cluster in self.clusters:
			for doc in cluster.members:
				true_label = doc.label
				for label in self.all_labels:
					if label == true_label:
						# actual positive
						if true_label == cluster.label:
							# True positive
							counts[label]['TP'] += 1
						else:
							# False negative
							counts[label]['FN'] += 1
					elif label == cluster.label:
						# predicted positive (TP already covered)
						counts[label]['FP'] += 1
					else:
						counts[label]['TN'] += 1
		return counts

	def calc_conf_mat(self):
		'''
		:return: {predicted_label : {actual_label : #occurances, }, }
		'''
		counts = {label: {label: 0 for label in self.all_labels} for label in self.all_labels}
		for cluster in self.clusters:
			for doc in cluster.members:
				counts[cluster.label][doc.label] += 1
		return counts


	def get_counts(self, label):
		return self.label_pr_mats[label].copy()

	def precision(self, label=None):
		try:
			counts = self.label_pr_mats[label] if label else self.total_counts
			return counts['TP'] / float(counts['TP'] + counts['FP'])
		except ZeroDivisionError:
			return 0.0

	def recall(self, label=None):
		try:
			counts = self.label_pr_mats[label] if label else self.total_counts
			return counts['TP'] / float(counts['TP'] + counts['FN'])
		except ZeroDivisionError:
			return 0.0

	def PR(self, label=None):
		return self.precision(label), self.recall(label)

	def F1(self, label=None):
		try:
			p, r = self.PR(label)
			return (2 * r * p) / (r + p)
		except ZeroDivisionError:
			return 0.0

	def F1_macro(self):
		return sum(map(lambda label: self.F1(label), self.all_labels)) / len(self.all_labels)

	def F1_micro(self):
		return self.F1(label=None)
        
	def majority_labels(self):
		return map(lambda x: self.majority_label(x), self.clusters)

	def majority_label(self, cluster):
		labels = map(lambda doc: doc.label, cluster.members)
		c = collections.Counter(labels)
		return c.most_common(1)[0][0]

	def accuracy(self):
		total_docs = 0
		correct = 0
		for cluster in self.clusters:
			total_docs += len(cluster.members)
			correct += len(filter(lambda doc: doc.label == cluster.label, cluster.members))
		return float(correct) / total_docs

	def label_entropy(self):
		label_entropy = 0.0
		label_counts = self.get_true_doc_histogram()
		for label in self.all_labels:
			prob = eps + label_counts[label] / float(self.num_docs)  # in the original paper, this is different
			label_entropy += prob * math.log(prob)
		label_entropy *= -1
		return label_entropy

	def cluster_entropy(self):
		cluster_entropy = 0.0
		for cluster in self.clusters:
			prob = eps + len(cluster.members) / float(self.num_docs)  # in the original paper, this is different
			cluster_entropy += prob * math.log(prob)
		cluster_entropy *= -1
		return cluster_entropy

	def v_measure(self):
		h = self.homogeneity()
		c = self.completeness()
		return utils.harmonic_mean(h, c)

	def completeness(self):
		if len(self.clusters) == 1:
			return 1.0

		# H(K|C)
		num = 0.0
		label_counts = self.get_true_doc_histogram()
		for label in self.all_labels:
			for cluster in self.clusters:
				one = self.label_cluster_mat[label][cluster._id] / float(self.num_docs)
				two = self.label_cluster_mat[label][cluster._id] / float(label_counts[label])
				num += one * math.log(two + eps)
		num *= -1

		return num / self.cluster_entropy()

	def homogeneity(self):
		if len(self.all_labels) == 1:
			return 1.0

		# H(C|K)
		num = 0.0
		for cluster in self.clusters:
			for label in self.all_labels:
				one = self.label_cluster_mat[label][cluster._id] / float(self.num_docs)
				two = self.label_cluster_mat[label][cluster._id] / float(len(cluster.members))
				num += one * math.log(two + eps)
		num *= -1
		
		# H(C)

		return 1.0 - (num / self.label_entropy())

	def ari(self):
		contingency_table = []
		for cluster in self.clusters:
			row = []
			for label in self.all_labels:
				row.append(map(lambda doc: doc.label, cluster.members).count(label))
		# TODO: finish
				


