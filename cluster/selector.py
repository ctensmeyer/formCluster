#!/usr/bin/python

import sys
import doc
import utils
import math
import functools
import heapq
import random
import time


def logProb(y,x,base):
    if (x == 0):
        return 0
   
    tmp = float(x)/y
#    print "X:", x, "Y:", y, "x/y:", tmp
#    print "Log:", math.log10(tmp)
    return (tmp)*math.log(tmp,base)


def similarityList(clustering,document):

    return map(lambda x: len(x.members), clustering)
    #return map(lambda c: document.similarity(c.center), clustering)


#def entropy(clustering):
    
#    print "Clusters:", len(clustering)

#    entropies = []

#    for c,cluster in enumerate(clustering):
#        for d,_doc in enumerate(cluster.members):
#            similarities = similarityList(clustering, _doc)
#            total = sum(similarities)
#            print similarities 
#            entropy = -1*sum(map(lambda s: logProb(total, s, len(clustering)), similarities))
#            entropies.append([c,d,entropy])
#        
#        if(c%10 == 0):
#            print c, "Clusters Processed"
#
#    entropies.sort(key=(lambda x: x[2]), reverse=True)
#
#    return entropies

def leastConfidence(clustering,_doc,cluster):
    similarities = map(lambda cluster: _doc.similarity(cluster.center), clustering)

    return 1-(max(similarities)/sum(similarities))

def margin(clustering, _doc, cluster):
    similarities = map(lambda cluster: _doc.similarity(cluster.center), clustering)

    print similarities

    total = sum(similarities)
    top2 = heapq.nlargest(2,similarities)

    if(total == 0):
        return 1

    return 1-(top2[0]-top2[1])/total


def entropy(clustering, _doc, cluster):
    similarities = map(lambda cluster: _doc.similarity(cluster.center), clustering)

    #print similarities
    total = sum(similarities)
    entropy = -1*sum(map(lambda s: logProb(total, s, len(clustering)) ,similarities))

    return entropy


cutoff = 15

def streamSelector(clustering):
    
    selection = []

    for c,cluster in enumerate(clustering):
        print "Cluster:", c
        select= []
        while(True):
            d = random.randint(0,len(cluster.members)-1)
            _doc = cluster.members[d]

            conf = entropy(clustering, _doc, c)
         #   print "Cluster:", c, "Document:", d, "Confidence:", conf
            
            shouldAdd = False

            split = random.random()
            
            shouldAdd = (conf>split)

            if(len(cluster.members) <= cutoff):
                shouldAdd = True

            if(shouldAdd):
                print "Added", len(select), (c,d,conf)
                select.append((c,d,conf))
                if(len(select) >= cutoff or len(select) == len(cluster.members)):
                    break

        selection.append(select)
        total = sum(map(lambda x: x[2], select))
        print "Average Confidence:", total/len(select)
    return selection




def pseudoDistance(clustering):
#   for each cluster, returns the pairwise distances of every point to every point in the cluster

    distances = []
    total = 0.0
    count = 0
    
    startTime = time.time()
    for cluster in clustering:
    
      #  print "Cluster Size:", len(cluster.members)
        '''
        print "Starting True Similarities"
        startTime = time.time()
        trueSimilarities = map(lambda r: map(lambda c: 1-c, r) ,utils.pairwise(cluster.members, lambda x,y: x.similarity(y)))
        endTime = time.time()
        print "Done. Elapsed Time:", endTime-startTime 
        '''
        
        namedSims = map(lambda x: x.similarities_by_name(cluster.center).items(), cluster.members)
        distToCenter = map(lambda x: [1-i[1] for i in x], namedSims)
        euclid = utils.pairwise(distToCenter, lambda x,y: utils.euclideanDistance(x,y))
        
        distances.append(euclid)
        
    
    endTime = time.time()
    print "Done. Elapsed Time:", endTime-startTime
    
    return distances

def singleLink(distances, x,y):
    minimum = 999999
    found = False
    
    for i in x:
        for j in y:
            if ((not found) or distances[i][j] < minimum):
                minimum = distances[i][j]
                found = True
                
    
    return minimum

def completeLink(distances, x,y):
    maximum = 0
    found = False
    
    found = False
    
    for i in x:
        for j in y:
            if ((not found) or distances[i][j] > maximum):
                maximum = distances[i][j]
                found = True

    
    return maximum

def HAC(distances, k=10, func=singleLink):
    clusters = [[i] for i in range(len(distances))]
    
    while (len(clusters) > k):
        minimum = 999999
        minIdx = (-1,-1)

        for i in range(len(clusters)):
           for j in range(len(clusters)):
                if (i == j):
                   continue
                if (j < i):
                   continue
                
                
                dist = func(distances, clusters[i], clusters[j])
                if (minIdx == (-1,-1) or dist < minimum):
                   minimum = dist
                   minIdx = (i,j)
                              
       #print "(%s, %s) %f" % (repr(clusters[minIdx[0]]), repr(clusters[minIdx[1]]), minimum)
        clusters[minIdx[0]] += clusters[minIdx[1]]
        del clusters[minIdx[1]]
        #print clusters

    return clusters


def selectRepresentatives(distances, clusters):
    reps = []
    
    for cluster in clusters:
        minDist = 99999
        minIdx = -1
        for x in cluster:
            dist = 0.0
            for y in cluster:
                dist += distances[x][y]
            if(minIdx == -1 or dist < minDist):
                minDist = dist
                minIdx = x
        
        if(minIdx != -1):
            reps.append(minIdx)
        
    
    return reps

def selectWithHac(clustering, k=10):
    
    distances = pseudoDistance(clustering)
    representatives = []
    
    for c, cluster in enumerate(clustering):
        dist = distances[c]
    
        subClusters = HAC(dist, k, completeLink)
    
        repIndx = selectRepresentatives(dist, subClusters)
        
        reps = []
        
        for r in repIndx:
            reps.append(cluster.members[r])
    
        representatives.append(reps)
        
    return representatives
    #print dist
    
    
def main(args):

    if(len(args) != 2):
                print "Usage: mds.py clustering.pkl"
                print "     C is the cluster in clustering.pkl to display"           
                sys.exit(0)

    path = args[1]
                
    print "Loading"
    clustering = utils.load_obj(path)

    print  len(selectWithHac(clustering))

    #print streamSelector(clustering)

    #print entropy(clustering)


if __name__ == '__main__':
    main(sys.argv) 
