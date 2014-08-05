
import cluster
import selector
import utils
import sys


class Hierarchy:

    @classmethod
    def createHierarchy(cls, clustering):
	
	root = cls()
	
	psuedo = selector.selectWithHac(clustering)
	
	map(lambda x: root.addRepresentative(cls(x.center)), clustering)
	
	for i, child in enumerate(root.representatives):
	    child.addRepresentatives(map(cls,psuedo[i]))
	
	return root
	
    def __init__(self, center = None, representatives = None):
	self.center = center
	self.representatives = representatives
	
    def addRepresentative(self, child):
	if(self.representatives == None):
	    self.representatives = []
	    
	self.representatives.append(child)
	
    def addRepresentatives(self, children):
	if(self.representatives == None):
	    self.representatives = []
	    
	self.representatives += children
	
    def setCenter(self, center):
	self.center = center
	
    def __repr__(self):
	children = None
	if (self.representatives != None):
	    children = map(str, self.representatives)
	return repr(self.center) + "   " + repr(children)
    
    
def main(args):

    if(len(args) != 2):
                print "Usage: mds.py clustering.pkl"
                print "     C is the cluster in clustering.pkl to display"           
                sys.exit(0)

    path = args[1]
                
    print "Loading"
    clustering = utils.load_obj(path)

    print "Creating Hierarchy"
    repr(Hierarchy.createHierarchy(clustering))


if __name__ == '__main__':
    main(sys.argv) 