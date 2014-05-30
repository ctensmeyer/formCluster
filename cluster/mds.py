#!/usr/bin/python

import numpy as np
import sys
import utils
import doc
import driver
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances as dist

class shapeArray:

    def __init__(self, data):
        self.array=data
        self.shape=[len(data), len(data[0])]


def docReduction(docs,N=2):
    similarities = utils.pairwise(docs, lambda x,y: x.similarity(y))
    return reduction(similarities, N)


def reduction(simMat,N=2):

    #change similarity matrix into dissimilarity matrix
    dis = map(lambda x: map(lambda y: 1-y, x), simMat)
    #dis = dist(simMat)
    #dis = simMat

    #configure MDS to run 10 times. Also specify that data will be a dissimilarity matrix
    mds = MDS(n_components=N, n_init=10,max_iter=3000, metric=False, dissimilarity="precomputed")
    mat = np.array(dis)
    
    #Run MDS
    fit = mds.fit(mat)
    print "Stress:", fit.stress_

    return fit.embedding_

def main(args):
    if(len(args) != 1):
        print "Usage: mds.py C clustering.pkl"
        print "     C is the cluster in clustering.pkl to display"
        sys.exit(0)

    #C = int(args[1])
    #path = args[2]
    
    print "Loading"
    #clustering = utils.load_obj(path)

    #docs = clustering[C].members
    docs = doc.get_docs_nested(driver.get_data_dir("very_small"))

    print "Calculating Pairwise Similarities"
    similarities = utils.pairwise(docs, lambda x,y: x.similarity(y))

    #print "INITIAL SIMILARITIES:"
    #utils.print_mat(similarities)

    print "Starting MDS"
    pos = reduction(similarities)

    print "MDS:"
    utils.print_mat(pos)

if __name__ == '__main__':
    main(sys.argv) 
