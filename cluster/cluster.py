
import utils
import collections

class Cluster:
	
	def __init__(self, members, center, _id = None):
		self.members = members
		self.center = center
		self.label = self.center.label
		self._id = _id

	def get_center(self):
		return self.center

	def get_members(self):
		return self.members

	def get_label(self):
		return self.label


class TemplateSorter:
	
	def __init__(self, docs):
		self.docs = docs

	def _add_cluster(self, doc):
		template = doc.copy(len(self.clusters))
		template.label = None
		self.clusters.append(Cluster([doc], template))

	def go(self, epsilon=0.20, templates=None):
		if templates is None:
			templates = list()
		self.clusters = [Cluster(list(), template) for template in templates]

		for x, doc in enumerate(self.docs):
			if self.clusters:
				similarities = map(lambda cluster: doc.similarity(cluster.center), self.clusters)
				idx = utils.argmax(similarities)
				if similarities[idx] > (1.0 - epsilon):
					cluster_match = self.clusters[idx]
					cluster_match.center.aggregate(doc)
					cluster_match.members.append(doc)
				else:
					self._add_cluster(doc)
			else:
				self._add_cluster(doc)

			if x % 10 == 0:
				print "%d documents processed" % x

	def prune_clusters(self, min_size=5, isolate=False):
		odd_docs = list()
		clusters_to_remove = list()
		for cluster in self.clusters:
			if len(cluster.members) < min_size:
				odd_docs += cluster.members
				clusters_to_remove.append(cluster)
		for cluster in clusters_to_remove:
			self.clusters.remove(cluster)
		if odd_docs:
			if isolate:
				# make a single cluster of the oddballs
				template = odd_docs[0].copy(len(self.templates))
				template.label = None
				for doc in odd_docs[1:]:
					template.aggregate(doc)
				self.clusters.append(Cluster(odd_docs, template))
			else:
				# distribute oddballs to closest cluster
				for doc in odd_docs:
					similarities = map(lambda cluster: doc.similarity(cluster.center), self.clusters)
					idx = utils.argmax(similarities)
					cluster_match = self.clusters[idx]
					cluster_match.center.aggregate(doc)
					cluster_match.members.append(doc)
			

	def get_clusters(self):
		return self.clusters

	
class CheatingSorter:
	
	def __init__(self, docs):
		self.docs = docs

	def go(self):
		self.templates = {}
		self.assignments = collections.defaultdict(list)

		for x, doc in enumerate(self.docs):
			doc._load_check()
			true_label = doc.label
			if true_label in self.templates:
				# now sure how dicts handle mutable objects...
				template = self.templates[true_label]
				template.aggregate(doc)
				self.templates[true_label] = template
			else:
				template = doc.copy(true_label)
				self.templates[true_label] = template
			self.assignments[true_label].append(doc)
	
				
	def get_clusters(self):
		assert len(self.templates) == len(self.assignments)
		clusters = []
		for label in sorted(self.templates):
			cluster = Cluster(self.assignments[label], self.templates[label])
			clusters.append(cluster)
		return clusters
			
