
import collections
import json

_counts = ['TP', 'TN', 'FP', 'FN']

class ClusterAnalyzer:

	def __init__(self, clusters, cluster_labels=None):
		self.clusters = clusters
		self.all_labels = self.get_all_labels()
		if cluster_labels:
			self.cluster_labels = cluster_labels
		else:
			self.cluster_labels = self.majority_labels()

		self.conf_mats = self.calc_conf_mats()
		self.total_counts = {count: 0 for count in _counts}
		self.num_docs = sum(map(len, clusters))
		for label in self.conf_mats:
			for count in _counts:
				n = self.conf_mats[label][count]
				self.total_counts[count] += n
		self.print_assignments()

	def print_assignments(self):
		print "ALL DOCUMENT ASSIGNMENTS:"
		print "\tID\tMatch?\tTrue Label\tAssigned Label"
		print "\t-----------------------------------------------------------"
		for cluster, label in zip(self.clusters, self.cluster_labels):
			print
			for doc in cluster:
				print "\t" + "\t".join([str(doc._id), str(doc.label == label), doc.label, label])
		print
		print

	def get_all_labels(self):
		labels = set()
		for cluster in self.clusters:
			for doc in cluster:
				labels.add(doc.label)
		return labels

	def get_true_doc_histogram(self):
		counts = collections.defaultdict(int)
		for cluster in self.clusters:
			for doc in cluster:
				counts[doc.label] += 1
		return counts

	def get_doc_histogram(self):
		counts = collections.defaultdict(int)
		for cluster, label in zip(self.clusters, self.cluster_labels):
			counts[label] += len(cluster)
		return counts

	def get_cluster_histogram(self):
		histo = collections.Counter(self.cluster_labels)
		histo.update(self.all_labels)
		histo.subtract(self.all_labels)
		return histo
	
	def print_all(self):
		print
		self.print_general_info()
		self.print_histogram_info()
		self.print_label_info()
		self.print_metric_info()
		print

	def print_label_info(self):
		print "LABEL INFO:"
		print "\tThere are %d true labels" % len(self.all_labels)
		print "\tThere are %d assigned labels" % len(set(self.cluster_labels))
		print "\t%d labels were not assigned" % (len(self.all_labels) - len(set(self.cluster_labels)))
		print "\n\tAssigned Labels\n\t-------------------------"
		for label in sorted(set(self.cluster_labels)):
			print "\t%s" % label
		print "\n\tTrue Labels not assigned\n\t-------------------------"
		for label in sorted(self.all_labels - set(self.cluster_labels)):
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
		print "\tF1 Macro: ", self.F1_macro()
		print "\tF1 Micro: ", self.F1_micro()
		print "\tTotal PR/RC: ", self.PR()
		print "\tLabel break down:"
		for label in self.all_labels:
			print "\n\t\t%s:" % label
			print "\t\tF1: %.3f" % self.F1()
			print "\t\tPR: %.3f\tRC: %.3f" % (self.PR(label))
			s = "\t".join(map(lambda count: "%s: %d (%2.1f%%)" % (count, self.conf_mats[label][count], 100.0 * self.conf_mats[label][count] / self.num_docs), _counts))
			print "\t\t%s" % s
		print
		print

	def calc_conf_mats(self):
		'''
		:return: {label : { count_type (TP, etc) : #occurances, }, }
		'''
		counts = {label: {count: 0 for count in ['TP', 'TN', 'FP', 'FN']} for label in self.all_labels}
		for cluster, assigned_label in zip(self.clusters, self.cluster_labels):
			for doc in cluster:
				true_label = doc.label
				for label in self.all_labels:
					if label == true_label:
						# actual positive
						if true_label == assigned_label:
							# True positive
							counts[label]['TP'] += 1
						else:
							# False negative
							counts[label]['FN'] += 1
					elif label == assigned_label:
						# predicted positive (TP already covered)
						counts[label]['FP'] += 1
					else:
						counts[label]['TN'] += 1
		return counts


	def get_counts(self, label):
		return self.conf_mats[label].copy()

	def precision(self, label=None):
		try:
			counts = self.conf_mats[label] if label else self.total_counts
			return counts['TP'] / float(counts['TP'] + counts['FP'])
		except ZeroDivisionError:
			return 0.0

	def recall(self, label=None):
		try:
			counts = self.conf_mats[label] if label else self.total_counts
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
		labels = map(lambda doc: doc.label, cluster)
		c = collections.Counter(labels)
		return c.most_common(1)[0][0]

	def accuracy(self):
		total_docs = 0
		correct = 0
		for cluster, label in zip(self.clusters, self.cluster_labels):
			total_docs += len(cluster)
			correct += len(filter(lambda doc: doc.label == label, cluster))
		return float(correct) / total_docs

	def ari(self):
		contingency_table = []
		for cluster in self.clusters:
			row = []
			for label in self.all_labels:
				row.append(map(lambda doc: doc.label, cluster).count(label))
		# TODO: finish
				


		 
