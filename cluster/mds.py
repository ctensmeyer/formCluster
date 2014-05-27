#!/usr/bin/python

import numpy as np
import sys
import utils
import doc
from sklearn.manifold import MDS


class shapeArray:

    def __init__(self, data):
        self.array=data
        self.shape=[len(data), len(data[0])]

def dimensionalityReduction(similarities):
    mds = MDS(n_components=2)

    shape = (len(similarities), len(similarities[0]))

    sim = np.array(similarities)

    
    fit = mds.fit(sim)
    bestFit = fit.embedding_
    bestStress = fit.stress_
    print "initial stress:", bestStress  
    for i in range(10):
        fit = mds.fit(sim)
        if (fit.stress_ < bestStress):
            bestFit = fit.embedding_
            bestStress = fit.stress_
            print "updating MDS"


    print "final stress:", bestStress
    return bestFit

def main(args):
    if(len(args) != 3):
        print "Usage: mds.py C clustering.pkl"
        print "     C is the cluster in clustering.pkl to display"
        sys.exit(0)

    C = int(args[1])
    path = args[2]
    
    clustering = utils.load_obj(path)

    cluster = clustering[C]

    similarities = utils.pairwise(cluster.members, lambda x,y: x.similarity(y))

    print "INITIAL SIMILARITIES:"
    utils.print_mat(similarities)

    pos = dimensionalityReduction(similarities)

    print "MDS:"
    utils.print_mat(pos)

if __name__ == '__main__':
    main(sys.argv) 
