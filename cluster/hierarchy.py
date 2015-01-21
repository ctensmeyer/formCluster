
import cluster
import selector
import utils
import sys
import mds

class Hierarchy:

    nextId = [0]
    
    @classmethod
    def createHierarchy(cls, clustering):
	
	root = cls()
	
	psuedo = selector.selectWithHac(clustering)
	
	map(lambda x: root.addRepresentative(cls(x.center)), clustering)
	
	for i, child in enumerate(root.representatives):
	    child.addRepresentatives(map(cls,psuedo[i]))
	
	return root
	
    def __init__(self, center = None, representatives = None, uId = None):
	self.center = center
	self.representatives = representatives
	self.centerPos = None
	self.mdsPos = None
	self.simMat = None
	
	if(uId == None):
	    self.uId = self.nextId[0]
	    self.nextId[0] += 1
	else:
	    self.uId = uId
	
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
	
	
    def reduce(self, classic=False, dim=2):
	if(self.representatives == None):
	    return
	
	docs = map(lambda x: x.center, self.representatives)
    
	if (self.center != None):
	    docs.append(self.center)
	
	if (self.simMat == None):
	    self.simMat = utils.pairwise(docs, lambda x,y: x.similarity(y))
	
	points = []
	if(classic):
	    points = mds.classicMDS(self.simMat,dim)
	else:
	    points = mds.reduction(self.simMat,dim)
	    
	if (self.center != None):    
	    self.centerPos = points[-1]
	    self.mdsPos = points[:-1]
	else:
	    self.mdsPos = points[:]
	    
	map(lambda x: x.reduce(classic,dim), self.representatives)
    
	
    def __repr__(self):
	children = None
	if (self.representatives != None):
	    children = map(str, self.representatives)
	return repr(self.centerPos) + "   " + repr(self.mdsPos) + "          " + str(children)
    
    
def main(args):

    if(len(args) != 2):
                print "Usage: mds.py clustering.pkl"
                print "     C is the cluster in clustering.pkl to display"           
                sys.exit(0)

    path = args[1]
                
    print "Loading"
    clustering = utils.load_obj(path)

    print "Creating Hierarchy"
    h = Hierarchy.createHierarchy(clustering)
    
    mds.hierarchyReduction(h)
    
    print repr(h)


if __name__ == '__main__':
    main(sys.argv) 