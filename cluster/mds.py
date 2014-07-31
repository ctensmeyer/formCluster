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


def getMat(value, n):
    return [[value] * n for i in range(n)]


def stress(D, X, distance=utils.euclideanDistance):
    
    denominator = 0.0
    numerator = 0.0
    
    for i in range(len(D)):
        for j in range(i,len(D)):
            d = D[i][j]
            f = distance(X[i],X[j])
            numerator += (f- d) ** 2
            denominator += d ** 2
    return np.sqrt(numerator/denominator)

def classicMDS(simMat, dim = 2):
    dis = map(lambda x: map(lambda y: 1-y, x), simMat)
    
    D = np.array(dis)
    
    
    n = len(D)
    Ones = np.array(getMat(1, n))
    
    I = np.array(getMat(0,n))
    
    for x in range(n):
        I[x][x] = 1


    J = np.matrix(I - (1/float(n))*Ones)
    
    P = np.matrix(D*D)
    
    B = -0.5 * J * P * J
    

    W,V = np.linalg.eigh(B)

    top = [0] * dim
    
    w = W[:]
    
    w = sorted(w, reverse=True)
    

    idx = np.where(W == w[0])
    vectors = V[:,idx[0]]
    values = W[idx]
    
    for i in range(1,dim):
        idx = np.where(W == w[i])
        vectors = np.append(vectors, V[:,idx[0]], axis = 1)
        values = np.append(values, W[idx])
            
        
        

    
    L = np.array(getMat(0.0,dim))
    for i in range(dim):
        L[i,i] = values[i]
        
    L = np.sqrt(L)


    X = np.array(vectors * L)
    
    print "Stress:", stress(dis,X)
    
    return X

def reduction(simMat,N=2):

    #change similarity matrix into dissimilarity matrix
    dis = map(lambda x: map(lambda y: 1-y, x), simMat)
    #dis = dist(simMat)
    #dis = simMat

    #configure MDS to run 10 times. Also specify that data will be a dissimilarity matrix
    mds = MDS(n_components=N, n_init=10,max_iter=3000, metric=True, dissimilarity="precomputed")
    mat = np.array(dis)
    
    #Run MDS
    fit = mds.fit(mat)
    print "Approximate Stress:", fit.stress_
    print "Stress:", stress(dis, fit.embedding_)

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
    docs = doc.get_docs_nested(driver.get_data_dir("small"))

    print "Calculating Pairwise Similarities"
    similarities = utils.pairwise(docs, lambda x,y: x.similarity(y))

    #print "INITIAL SIMILARITIES:"
    #utils.print_mat(similarities)

    #similarities = [[0,93,82,133],[93,0,52,60],[82,52,0,111],[133,60,111,0]]

    print "Starting MDS"
    #pos = reduction(similarities)
    pos = classicMDS(similarities)

    print "MDS:"
    utils.print_mat(pos)

if __name__ == '__main__':
    main(sys.argv) 
