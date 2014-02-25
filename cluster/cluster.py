
import utils

class TemplateSorter:
	
	def __init__(self, docs):
		self.docs = docs

	def _add_template(self, doc):
		template = doc.copy(len(self.templates))
		self.templates.append(template)
		self.references.append([doc])

	def go(self, epsilon, templates=None):
		if templates is None:
			templates = list()
		self.templates = templates
		self.references = [list() for template in self.templates]

		for x, doc in enumerate(self.docs):
			if self.templates:
				distances = map(lambda template: doc.text_line_distance(template), self.templates)
				idx = utils.argmax(distances)
				if distances[idx] < (1.0 - epsilon):
					self._add_template(doc)
				else:
					template_match = self.templates[idx]
					template_match.aggregate(doc)
					self.references[template_match._id].append(doc)
			else:
				self._add_template(doc)

			if x % 10 == 0:
				print "%d documents processed" % x

	def get_clusters(self):
		return self.references

	def get_templates(self):
		return zip(self.templates, self.references)
		
