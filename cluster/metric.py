
import collections
import datetime
import json
import utils
import math
import sys
import os

import sklearn.metrics

_counts = ['TP', 'TN', 'FP', 'FN']
eps = 10e-10
_output_dir = "output/"

class KnownClusterAnalyzer:

	def __init__(self, confirm):
		self.confirm = confirm
		self.clusters = confirm.get_clusters()
		self.preprocess_clusters()

		self.docs = utils.flatten(map(lambda cluster: cluster.members, self.clusters))
		self.all_labels = self.get_all_labels()
		self.num_docs = len(self.docs)

		self.label_pr_mats = self.calc_label_pr_mats()
		self.label_cluster_mat = self.calc_label_cluster_counts()
		self.total_counts = {count: 0 for count in _counts}
		for label in self.label_pr_mats:
			for count in _counts:
				n = self.label_pr_mats[label][count]
				self.total_counts[count] += n
		#print json.dumps(self.label_pr_mats, indent=4)

		labels = list(self.all_labels)
		mapping = {label: labels.index(label) for label in labels}
		self.true_labels = map(lambda _doc: mapping[_doc.label], self.docs)
		self.predicted_labels = list()
		for _doc in self.docs:
			for _cluster in self.clusters:
				if _doc in _cluster.members:
					self.predicted_labels.append(mapping[_cluster.label])
					break
		#self.predicted_labels = utils.flatten(map(lambda cluster: ([mapping[cluster.label]] * len(cluster.members)) 
		#		if cluster.label else [], self.clusters))


	def preprocess_clusters(self):
		self.clusters.sort(key=lambda cluster: len(cluster.members), reverse=True)
		num = 0
		for cluster in self.clusters:
			if cluster._id is None:
				cluster._id = num
				num += 1
			cluster.set_label()

	def draw_centers(self):
		# make the appropriate directory
		dir_name = os.path.join(_output_dir, str(datetime.datetime.now()).replace(' ', '_') + "_".join(sys.argv[1:]))
		try:
			os.mkdir(dir_name)
		except:
			pass
		
		# write the arguments so we know what we did
		f = open(os.path.join(dir_name, "args.txt"), 'w')
		f.write(repr(sys.argv))
		f.close()

		# save clusters to file
		utils.save_obj(self.confirm, os.path.join(dir_name, "confirm.pkl"))		

		# draw each cluster center
		for cluster in self.clusters:
			im = cluster.center.draw()
			im.save(os.path.join(dir_name, "cluster_%d.png" % cluster._id))
			

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
		for _doc in self.docs:
			labels.add(_doc.label)
		return labels

	def get_true_doc_histogram(self):
		counts = collections.defaultdict(int)
		for _doc in self.docs:
			counts[_doc.label] += 1
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
		#self.print_doc_cluster_sim_mat()
		#self.print_cluster_mismatches()
		self.print_feature_eval()
		self.print_cluster_separation()
		self.print_region_info()
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


	def print_cluster_sim_mat(self):
		print "CLUSTER SIM MATRICES:"

		centers = map(lambda cluster: cluster.center, self.clusters)
		feature_set_names = self.docs[0].get_feature_set_names()
		for name in feature_set_names:
			print
			print "Similarity Type: %s" % name
			mat = utils.pairwise(centers, lambda doc1, doc2: doc1.global_sim(doc2, name))
			mat = utils.apply_mat(mat, lambda x: "%3.2f" % x)
			utils.insert_indices(mat)
			utils.print_mat(mat)

		print
		print "Similarity Type: Cluster sim by CONFIRM"
		sub_mat = utils.pairwise(self.clusters, lambda c1, c2: self.confirm.cluster_similarity(c1, c2))
		sub_mat = utils.apply_mat(sub_mat, lambda x: "%3.2f" % x)
		utils.insert_indices(sub_mat)
		utils.print_mat(sub_mat)
		print
		print

	def print_cluster_mismatches(self):
		print "CLUSTER MISMATCHES"
		print
		total = 0
		for x, cluster in enumerate(self.clusters):
			print "%d:\t%s" % (x, cluster.label)
			for _doc in cluster.members:
				if _doc.label != cluster.label:
					total += 1
					#sims = cluster.center.global_sim(_doc)
					#sim = self.confirm.cluster_doc_similarity(cluster, _doc)
					#print "\t" + "\t".join(map(str, [_doc.label, sim, sims]))
					#print "\t" + "\t".join(map(str, [_doc.label, sim, sims]))
					print "\t%s" % _doc.label
			print
		print "Total Mismatches:", total
		print
		print

	def feature_eval_metrics(self, sim_fun):
		doc_cluster_sims_flat = list()
		doc_cluster_means = list()
		doc_cluster_std_devs = list()
		for cluster in self.clusters:
			cluster_sims = list()
			for _doc in cluster.members:
				val = sim_fun(cluster, _doc)
				doc_cluster_sims_flat.append(val)
				cluster_sims.append(val)
			doc_cluster_means.append(utils.avg(cluster_sims))
			doc_cluster_std_devs.append(utils.stddev(cluster_sims))
		global_mean = utils.avg(doc_cluster_sims_flat)
		global_stddev = utils.stddev(doc_cluster_sims_flat)
		mean_of_means = utils.avg(doc_cluster_means)
		stddev_of_means = utils.stddev(doc_cluster_means)
		mean_of_stddev = utils.avg(doc_cluster_std_devs)
		stddev_of_stddev = utils.stddev(doc_cluster_std_devs)

		return global_mean, global_stddev, mean_of_means, stddev_of_means, mean_of_stddev, stddev_of_stddev
		

	def print_feature_eval(self):
		'''
		For each feature type, calculate the means and the std dev for each cluster.  Then
			take the mean and std dev of those quantaties
		'''
		print "FEATURE EVAL"
		print

		stats = dict()

		for name in self.clusters[0].members[0].get_feature_set_names():
			stats[name + "_global"] = self.feature_eval_metrics(
				lambda cluster, _doc: cluster.center.global_sim(_doc, name))
			stats[name + "_region_uniform_weights"] = self.feature_eval_metrics(
				lambda cluster, _doc: utils.avg_val_mat(cluster.center.region_sim(_doc, name)))
			stats[name + "_region_fixed_weights"] = self.feature_eval_metrics(
				lambda cluster, _doc: utils.avg_val_mat(utils.mult_mats(cluster.center.region_sim_weights(_doc, name))))


		stats['confirm'] = self.feature_eval_metrics(
			lambda cluster, _doc: self.confirm.cluster_doc_similarity(cluster, _doc))

		padding_len = 1 + max(map(len, stats.keys()))
		print "Columns are in order:"
		print "1) Mean Similarity between Document and assigned Clusters"
		print "2) Std. Dev of Document-Cluster similarities"
		print "3) Mean Average Cluster similarity (cluster members to cluster)"
		print "4) Std Dev of Average Cluster similarity"
		print "5) Mean of Std Dev of Cluster similarity (cluster members to cluster)"
		print "6) Std Dev of the Std Dev of Cluster similarity"
		print
		for name in sorted(list(stats.keys())):
			print utils.pad_to_len(name, padding_len), "\t", "\t".join(map(lambda x: "%.4f" % x, stats[name]))
			if "_uniform" in name or "overall" in name:
				print

		print
		print

	def print_cluster_separation(self):
		print "CLUSTER SEPERATION"
		print
		print "Comparing each Cluster to it's most similar other clusters"

		if len(self.clusters) < 2:
			print "There are less than two clusters"
			return

		cluster_sim_mat = self.confirm.get_cluster_sim_mat()
		for row in cluster_sim_mat:
			row.sort(reverse=True)

		top_1 = list()
		top_3 = list()
		top_5 = list()
		for row in cluster_sim_mat:
			for x, val in enumerate(row):
				if x == 0:
					continue
				if x <= 1:
					top_1.append(val)
				if x <= 3:
					top_3.append(val)
				if x <= 5:
					top_5.append(val)
				else:
					break
		top_1.sort(reverse=True)

		top_1_mean = utils.avg(top_1)
		top_1_stddev = utils.stddev(top_1)
		top_3_mean = utils.avg(top_3)
		top_3_stddev = utils.stddev(top_3)
		top_5_mean = utils.avg(top_5)
		top_5_stddev = utils.stddev(top_5)
		print "\n        Mean\t   Std Dev"
		print "Top 1: %3.3f\t %3.3f" % (top_1_mean, top_1_stddev)
		print "Top 3: %3.3f\t %3.3f" % (top_3_mean, top_3_stddev)
		print "Top 5: %3.3f\t %3.3f" % (top_5_mean, top_5_stddev)
		print
		print "List of 10 most similar scores"
		print ", ".join(map(lambda x: "%4.3f" % x, top_1[:10]))

		print
		print

	def print_cluster_cohesion(self):
		print "CLUSTER COHESION:"
		sim_names = self.clusters[0].members[0].get_feature_set_names()[:]
		sim_names.append("confirm")
		print "\t\t%s     SIZE" % ("        ".join(sim_names))
		for x, cluster in enumerate(self.clusters):
			# list of lists
			similarities = map(lambda _doc: _doc.global_sim(cluster.center), cluster.members)
			to_print = list()
			for y in xrange(len(similarities[0])):
				values = map(lambda row: row[y], similarities)
				to_print.append(utils.avg(values))
				to_print.append(utils.stddev(values))
			values = map(lambda _doc: self.confirm.cluster_doc_similarity(cluster, _doc), cluster.members)
			to_print.append(utils.avg(values))
			to_print.append(utils.stddev(values))
			l = len(cluster.members)
			print "\t%s:  %s  %d" % (x, "  ".join(map(lambda s: "%3.2f" % s, to_print)), l)
		print
		print

	def print_doc_cluster_sim_mat(self):
		print "CLUSTER-DOC SIM MAT"
		print
		for x, cluster in enumerate(self.clusters):
			print "%d:\t%s" % (x, cluster.label)


		print
		print "documents labeled with # indicate that their most similar cluster has a different true label"
		print "documents labeled with ^ indicate that their assigned cluster is not the most similar cluster"
		print "cluster sim scores labeled with * indicate that the cluster shares the label with the document"
		print

		print (" " * 50) + "\t\t".join(map(str, xrange(len(self.clusters))))
		print

		num_closest_to_incorrect_cluster = 0
		doc_cluster_sim_mat = self.confirm.get_doc_cluster_sim_mat()
		for doc_idx in xrange(len(self.docs)):
			_doc = self.docs[doc_idx]
			to_print = list()
			best_cluster = None
			best_sim_score = -1
			post = ""
			for cluster_idx in xrange(len(self.clusters)):
				cluster = self.clusters[cluster_idx]
				sim_score = doc_cluster_sim_mat[doc_idx][cluster_idx]
				if sim_score > best_sim_score:
					best_cluster = cluster
					best_sim_score = sim_score
				to_print.append("%3.2f" % sim_score)
				if (cluster.label == _doc.label):
					to_print[-1] += '*'
			if _doc.label != best_cluster.label:	
				num_closest_to_incorrect_cluster += 1
				post += "#"
			if _doc not in best_cluster.members:
				post += "^"
			print "%s%s" % (utils.pad_to_len("%s %s" % (_doc._id, (_doc.label + post)), 50), "\t".join(to_print))

		#for _doc in self.docs:
		#	to_print = list()
		#	best_cluster = None
		#	best_sim_score = -1
		#	post = ""
		#	for cluster in self.clusters:
		#		sim_score = self.confirm.cluster_doc_similarity(cluster, _doc)
		#		if sim_score > best_sim_score:
		#			best_cluster = cluster
		#			best_sim_score = sim_score
		#		to_print.append("%3.2f" % sim_score)
		#		if (cluster.label == _doc.label):
		#			to_print[-1] += '*'
		#	if _doc.label != best_cluster.label:	
		#		num_closest_to_incorrect_cluster += 1
		#		post += "#"
		#	if _doc not in best_cluster.members:
		#		post += "^"
		#	print "%s%s" % (utils.pad_to_len("%s %s" % (_doc._id, (_doc.label + post)), 50), "\t".join(to_print))

		print
		print "Number of docs most similar to a wrong cluster: %d / %d = %2.1f%%" % (
			num_closest_to_incorrect_cluster, len(self.docs), 100.0 * num_closest_to_incorrect_cluster / len(self.docs))
		print
		print

	def print_region_info(self):
		print "REGION INFO"
		print
		print "For each feature set, the average sim score per region by cluster and the fixed weights"
		print

		for x, cluster in enumerate(self.clusters):
			print "%d:\t%s\t%d\n" % (x, cluster.label, len(cluster.members))
			for name in cluster.center.get_feature_set_names():
				mats = map(lambda _doc: cluster.center.region_sim(_doc, name), cluster.members)
				avg_mat = utils.avg_mats(mats)
				weight_mat = cluster.center.region_weights(name)
				print name
				utils.print_mat(utils.apply_mat(avg_mat, lambda x: "%.3f" % x))
				print
				utils.print_mat(utils.apply_mat(weight_mat, lambda x: "%.3f" % x))
				print



			#list_of_sim_mats = map(lambda _doc: cluster.center.similarity_mats_by_name(_doc), cluster.members)
			#list_of_weight_mats = cluster.center.similarity_weights_by_name(cluster.members[0])
			#for name in cluster.center.similarity_function_names():
			#	mats = map(lambda x: x[name], list_of_sim_mats)
			#	avg_mat = utils.avg_mats(mats)
			#	weight_mat = list_of_weight_mats[name]
			#	print name
			#	utils.print_mat(utils.apply_mat(avg_mat, lambda x: "%.3f" % x))
			#	print
			#	utils.print_mat(utils.apply_mat(weight_mat, lambda x: "%.3f" % x))
			#	print
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
		print "\tARI: ", self.ari()
		print "\tNum Clusters: ", len(self.clusters)
		print "\tF1 Macro: ", self.F1_macro()
		print "\tF1 Micro: ", self.F1_micro()
		print "\tTotal PR/RC: ", self.PR()
		_total = sum(map(lambda count: self.total_counts[count], _counts))
		s = "\t".join(map(lambda count: "%s: %d (%2.1f%%)" % (count, self.total_counts[count],
					100.0 * self.total_counts[count] / _total), _counts))
		print "\tTotal Counts:", s
		print "\tLabel break down:"
		for label in self.all_labels:
			print "\n\t\t%s:" % label
			print "\t\tF1: %.3f" % self.F1(label)
			print "\t\tPR: %.3f\tRC: %.3f" % (self.PR(label))
			s = "\t".join(map(lambda count: "%s: %d (%2.1f%%)" % (count, self.label_pr_mats[label][count],
						100.0 * self.label_pr_mats[label][count] / self.num_docs), _counts))
			print "\t\t%s" % s
		print
		print

	def print_summary(self, k, s, prefix="init", sil=-1):
		acc = self.accuracy()
		v = self.v_measure()
		h = self.homogeneity()
		c = self.completeness()
		ari = self.ari()

		ints = [k, s]
		metrics = [acc, v, c, h, ari, sil]

		print
		print "%s\t%s\n" % (prefix, "\t".join(map(lambda x: "%d" % x, ints)) + "\t" +
									"\t".join(map(lambda x: "%.3f" % x, metrics)))

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
		#h = self.homogeneity()
		#c = self.completeness()
		#return utils.harmonic_mean(h, c)
		return sklearn.metrics.v_measure_score(self.true_labels, self.predicted_labels)

	def completeness(self):
		#if len(self.clusters) == 1:
		#	return 1.0

		## H(K|C)
		#num = 0.0
		#label_counts = self.get_true_doc_histogram()
		#for label in self.all_labels:
		#	for cluster in self.clusters:
		#		one = self.label_cluster_mat[label][cluster._id] / float(self.num_docs)
		#		two = self.label_cluster_mat[label][cluster._id] / float(label_counts[label])
		#		num += one * math.log(two + eps)
		#num *= -1

		#return 1.0 - (num / self.cluster_entropy())
		return sklearn.metrics.completeness_score(self.true_labels, self.predicted_labels)

	def homogeneity(self):
		#if len(self.all_labels) == 1:
		#	return 1.0

		## H(C|K)
		#num = 0.0
		#for cluster in self.clusters:
		#	for label in self.all_labels:
		#		one = self.label_cluster_mat[label][cluster._id] / float(self.num_docs)
		#		two = self.label_cluster_mat[label][cluster._id] / float(len(cluster.members))
		#		num += one * math.log(two + eps)
		#num *= -1
		#
		## H(C)

		#return 1.0 - (num / self.label_entropy())
		return sklearn.metrics.homogeneity_score(self.true_labels, self.predicted_labels)

	def ari(self):
		return sklearn.metrics.adjusted_rand_score(self.true_labels, self.predicted_labels)

