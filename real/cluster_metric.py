

class ClusterAnalyzer:

	def __init__(self, cluster_set, cluster_labels, all_labels):
		self.cluster_set = cluster_set
		self.cluster_labels = cluster_labels
		self.all_labels = all_labels
		self.conf_mats = self.calc_conf_mats(cluster_set, cluster_labels, all_labels)
		self.total_counts = {count: 0 for count in ['TP', 'TN', 'FP', 'FN']}
		for label in self.conf_mats:
			for count in ['TP', 'TN', 'FP', 'FN']:
				self.total_counts[count] += self.conf_mat[label][count]
		self.true_labels = self.majority_labels()

	def calc_conf_mats(self, cluster_set, cluster_labels, all_labels):
		counts = {label: {count: 0 for count in ['TP', 'TN', 'FP', 'FN']} for label in all_labels}
		for cluster, assigned_label in zip(cluster_set, cluster_labels):
			doc_labels = map(lambda doc: doc.label, cluster)
			c = collections.Counter(doc_labels)
			size = len(cluster)
			for label in all_labels:
				if label == assigned_label:
					continue
				counts[label]['TN'] += (size - c.get(label)) # everything that was assigned a different label
				counts[label]['FP'] += c.get(label) # everything that was assigned this label 
			counts[assigned_label]['TP'] += c.get(assigned_label)
			counts[assigned_label]['FN'] += (size - c.get(assigned_label))
		return counts

	def get_counts(self, label):
		return self.conf_mats[label].copy()

	def precision(self, label):
		counts = self.conf_mats[label] if label else self.total_counts
		return counts['TP'] / float(counts['TP'] + counts['FP'])

	def recall(self, label):
		counts = self.conf_mats[label] if label else self.total_counts
		return counts['TP'] / float(counts['TP'] + counts['FN'])

	def PR(self, label):
		counts = self.conf_mats[label] if label else self.total_counts
		return (counts['TP'] / float(counts['TP'] + counts['FP']),
		counts['TP'] / float(counts['TP'] + counts['FN']))

	def F1(self, label):
		p, r = self.PR(label)
		return (2 * r * p) / (r + p)

	def F1_macro(self):
		return sum(map(lambda label: self.F1(label), self.all_labels)) / len(self.all_labels)

	def F1_micro(self):
		return self.F1(label=None)
        
	def majority_labels(self):
		return map(lambda x: self.majority_label(x), self.cluster_set)

	def majority_label(self, cluster):
		all_labels = map(lambda doc: doc.label, cluster)
		c = collections.Counter(all_labels)
		return c.most_common(1)[0][0]

	def accuracy(self):
		total_docs = 0
		correct = 0
		for cluster, true_label in zip(self.cluster_set, self.true_labels):
			total_docs += len(cluster)
			correct += len(filter(lambda doc: doc.label == true_label, cluster))
		return float(correct) / total_docs

	def ari(self):
		contingency_table = []
		for cluster in self.cluster_set:
			row = []
			for label in self.all_labels:
				row.append(map(lambda doc: doc.label, cluster).count(label))
				


		 
