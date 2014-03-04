
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

	def _add_template(self, doc):
		template = doc.copy(len(self.templates))
		template.label = None
		self.templates.append(template)
		self.assignments.append([doc])

	def go(self, epsilon=0.20, templates=None):
		if templates is None:
			templates = list()
		self.templates = templates
		self.assignments = [list() for template in self.templates]

		for x, doc in enumerate(self.docs):
			if self.templates:
				distances = map(lambda template: doc.text_line_distance(template), self.templates)
				idx = utils.argmax(distances)
				if distances[idx] < (1.0 - epsilon):
					self._add_template(doc)
				else:
					template_match = self.templates[idx]
					template_match.aggregate(doc)
					self.assignments[template_match._id].append(doc)
			else:
				self._add_template(doc)

			if x % 10 == 0:
				print "%d documents processed" % x

	def get_clusters(self):
		assert len(self.templates) == len(self.assignments)
		clusters = []
		for x in xrange(len(self.templates)):
			cluster = Cluster(self.assignments[x], self.templates[x])
			clusters.append(cluster)
		return clusters

	
class CheatingSorter:
	
	def __init__(self, docs):
		self.docs = docs

	def go(self):
		self.templates = {}
		self.assignments = collections.defaultdict(list)

		for x, doc in enumerate(self.docs):
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
			cluster = Cluster(self.assignments[x], self.templates[x])
			clusters.append(cluster)
		return clusters
			
