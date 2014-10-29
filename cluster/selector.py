#!/usr/bin/python

import sys
import doc
import utils
import math
import functools
import heapq
import random
import time
import heapq
import itertools
import numpy as np
from cluster import Cluster
from cluster import BaseCONFIRM
import metric 
import matplotlib.pyplot as pyplot
from image.signalutils import blur_bilateral


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
#   for each cluster, returns the estimated pairwise distances of every point to every point in the cluster

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
        
        #Order Matters here.
        '''
        namedSims = map(lambda x: cluster.center.similarities_by_name(x).items(), [cluster.center] + cluster.members)
        distToCenter = map(lambda x: [1-i[1] for i in x], namedSims)
        euclid = utils.pairwise(distToCenter, lambda x,y: utils.euclideanDistance(x,y))
        
        distances.append(euclid)
        '''
        
        vectors = map(lambda x: cluster.center.similarity_vector(x), [cluster.center] + cluster.members)
        distToCenter = map(lambda x: map(lambda i: 1-i, x), vectors)
        euclid = utils.pairwise(distToCenter, lambda x,y: utils.euclideanDistance(x,y))
        
        distances.append(euclid)
    
    endTime = time.time()
    print "Done. Elapsed Time:", endTime-startTime
    
    return distances

#######################################################################################################################################

class heap:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()
        #print "Init"
        
    def push(self, task, priority =0):
        'Add a new task or update the priority of an existing task'
        #print "Pushing"
        if task in self.entry_finder:
            self.remove(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)
        
    def remove(self,task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(task)
        entry[-1] = self.REMOVED
        
    
    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
    
    def isEmpty(self):
        return len(self.entry_finder) == 0

class dataPoint:
    def __init__(self, _id):
        self._id = _id
        self.processed = False
        self.reachability = -1
        self.core = -1
        
    def __eq__(self, other):
        return self._id == other._id
    
    def core_distance(self, distances, minPts):
        if(self.core == -1):
            d = sorted(distances)
            self.core = d[ minPts ] if (minPts < len(d)) else -1
        
        return self.core
    
    def __hash__(self):
        return hash(self._id)


def OPTICS_update(distances, N, p, seeds, minPts):
    coredist = p.core_distance(distances, minPts)
    for o in N:
        if(not o.processed):
            new_reach = max(coredist, distances[o._id])
            if(o.reachability == -1):
                o.reachability = new_reach
                seeds.push(o, new_reach)
            elif (new_reach < o.reachability):
                o.reachability = new_reach
                seeds.push(o, new_reach)

                

def OPTICS(distances, minPts, cluster=None):
    
    points = map(dataPoint, range(len(distances)))
    
    #random.shuffle(points)
    output = []
    
    for p in points:
        if p.processed:
            continue
        
        N = points[:]
        N.remove(p)
        p.processed = True
        output.append(p)
        
        if(p.core_distance != -1):
            seeds = heap()
            OPTICS_update(distances[p._id], N, p, seeds, minPts)
            while (not seeds.isEmpty()):
                q = seeds.pop()
                q.processed = True
                output.append(q)
                if(q.core_distance(distances[q._id], minPts) != -1):
                    OPTICS_update(distances[q._id], N, q, seeds, minPts)
        
    output[0].reachability = 0
    return output

    #return chooseMax(output)

def displayTruePlot(output, cluster, minPts, display=9):
    labels = [cluster.center] + map(lambda x: x.label, cluster.members)
    ids = map(lambda x: x._id, output)
    reach = map(lambda x: x.reachability, output)
    reach[0] = 0
    
    colors = ['r^', 'g^', 'm^', 'c^','b^', 'y^', 'k^']
    color = {}
    c = 0
    
    for lbl in labels:
        if lbl not in color:
            color[lbl] = colors[c % len(colors)]
            c += 1
    
    
    
    fig = pyplot.figure(1)
    pyplot.clf()
    
    pyplot.title("True Values")
    
    pyplot.plot(reach, "k")
    
    l = zip(ids,reach)
    x = 1
    
    centerColor = 'k^' if (cluster.label == None) else color[cluster.label]
    pyplot.plot(0,0, centerColor)
    
    for i, r in l[1:]:
        pyplot.plot(x, r, color[cluster.members[i-1].label])
        x+=1
    
    minimum = selectMinimum(output,minPts)
    
    for m in minimum[:display-1]:
        pyplot.plot(output.index(m), m.reachability, 'kD')
    
    fig.show()
    
    
def selectMinimum(output, minPts):
    
    reachability = np.array(map(lambda x: x.reachability, output))
    fEdge = np.array((0, 1, -1))
    
    reach = reachability
    kernel = fEdge
    forward = np.correlate(reach, kernel, mode='same')
    
    minimum = []
    assert(len(forward) == len(output))
    
    dist = minPts*2
    for i,o in enumerate(output):
        start = i - dist
        end = i
        
        if (start < 0):
            start = 0 

        

        threshold = 0.0
        if (((i != 0) and (i != len(output))) and ((forward[i] < threshold) and (forward[i-1] >= threshold))):
            area = reachability[start:end]
            std = max(area)
            minimum.append((std - reachability[i], i))

    minimum.sort()
    
    minimum.reverse()
    
    ret = map(lambda i: output[i[1]], minimum)
    
    return ret

def separateClusters(output, minPts):
    output[0].reachability = 0
    reachability = np.array(map(lambda x: x.reachability, output))
    fEdge = np.array((0, 1, -1))
    avg = np.array([1/float(minPts)] * minPts)


    space_sig = 1
    value_sig = 0.001
    
    reach = reachability
    #reach = blur_bilateral(reachability, minPts, space_sig, value_sig)
    
    
    kernel = fEdge
    #kernel = np.correlate(avg, fEdge, mode='full')
    
    forward = np.correlate(reach, kernel, mode='same')

    std = np.std(forward)
    
    fig = pyplot.figure(3)
    pyplot.clf()
    pyplot.title("Derivative")
    
    pyplot.plot(forward, color='g')   
    

    clusters = []
    current = []
    dist = minPts*2
    for i, o in enumerate(output):
        
        start = i - dist
        end = i + dist
        
        if (start < 0):
            start = 0 
        if (end > (len(reach)-1)):
            end = len(reach) -1
        
        area = reach[start:end]
        
        var = np.std(area)
        
        threshold = np.sqrt(var * std)
        pyplot.plot(i,std, 'c.')
        pyplot.plot(i,threshold, 'b.')
        if (((i == 0) and (i != len(output)-1)) or ((forward[i] >= threshold) and (forward[i-1] < threshold))):
            current = []
            clusters.append(current)
            
    
        current.append(o)
          
    
    fig.show()

    while(len(clusters[0]) < minPts and len(clusters) > 1):
        if (len(clusters[0]) < minPts and len(clusters) > 1):
            clusters[1] = clusters[0] + clusters[1]
            del clusters[0]
    
    enum = reversed(zip(range(len(clusters)), clusters))
    
    for i,cl in enum:
        if (i ==0):
            continue
        
        if(len(cl) < minPts):
            clusters[i-1] += cl
            del clusters[i]



    fig = pyplot.figure(2)
    pyplot.clf()
    
    pyplot.title("Clustering")
    
    pyplot.plot(reachability, "c")
    pyplot.plot(reach, "k")
    
    colors =['b^','r^', 'g^', 'm^', 'c^']
    c = 0

    i = 0
    for cl in clusters:
        c = (c+1) % len(colors)
        for o in cl:
            pyplot.plot(i,o.reachability, colors[c])
            i += 1
    fig.show()
    
    print "clusters:", len(clusters), map(lambda x: len(x), clusters)
    
    #print "variance:", np.var(reach),",", np.var(forward)
    
    #raw_input()
    
    return clusters

def chooseMax(output):
    reach =  map(lambda x: x.reachability, output)

    reach[0] = 0.0

    s = zip(reach, range(len(reach)))
    s.sort()
        
    m = s[-8:]
    
    reps, idx = zip(*m)

    fig = pyplot.figure(2)
    pyplot.clf()
    
    
    pyplot.plot(reach, color='b')   
    pyplot.plot(idx, reps, "g^")
    fig.show()
    
    idx = [0] + sorted(idx)
    
    return map(lambda i: output[i]._id, idx)

#################################################################################################################################
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
    
    

def displayPoints(_docs):
    
    pyplot.clf()
    
    rows = 3
    cols = 3
    
    fig = pyplot.figure(4)
    pyplot.clf()
    
    for i,_doc in enumerate(_docs):
        if (i >= rows*cols):
            break
        
        
        im = _doc.draw()
        
        pyplot.subplot(rows,cols,i+1)
        
        pyplot.axis('off')
        pyplot.tight_layout(pad=0.0, w_pad=0.1, h_pad=0.1)
        
        pyplot.imshow(im, interpolation="bicubic")
        #print "Displaying %d" % (i+1)
    
    mng = pyplot.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    
    fig.show()
    #raw_input()
    
    
def validate(clusters, display=False):
    
    purities = []
    for i,cluster in enumerate(clusters):
        labels = {}
        
        for _doc in cluster.members:
            lbl = _doc.label
            if(lbl in labels):
                labels[lbl] += 1
            else:
                labels[lbl] = 1
                
                
        lblCount = max(labels.values())
        purity = lblCount/float(len(cluster.members))
        purities.append(purity)
        if display:
            print "Cluster %d of %d -- %f Pure" % (i, len(clusters), purity)
            print "    %d Labels Found" % (len(labels))
        
            for key in labels:
                print "        %s: %d" %(key, labels[key])
    
    return purities
    

#Count the Number of Queries presented to the user.
QueryCount = 0

def simulateUserFeedback(label, group):
    global QueryCount
    same = []
    different = []
    
    for _doc in group:
        if(_doc.label == label):
            same.append(_doc)
        else:
            different.append(_doc)
    
    QueryCount += 1
    return same, different

def mergeClusters(clusters, N):
    '''
        find N closest clusters to each cluster. Query user. 
    '''
    
    centers = map(lambda x: x.center, clusters)
    
    idx = {}
    revIdx = {}
    
    newClusters =[]
    
    merged = {}
    
    for i, c in enumerate(centers):
        idx[c] = i
    
    simMat = utils.pairwise(centers, lambda x,y: x.similarity(y), False)
    
    for c in centers:
        if c in merged:
            continue
        
        sims = np.array(simMat[idx[c]])
        #Find indecies of each of the top N points
        argmax = np.argsort(-sims)[:N-1]
        
        #Map indecies to centers
        _docs = [c] + map(lambda x: centers[x], argmax)
    
        label = clusters[idx[c]].label
        
        same,different = simulateUserFeedback(label, _docs)
        
        #print "Same:", len(same), "Different:", len(different)
        
        m = filter(lambda s: s in merged, same)
        mergeTo = merged[m[0]] if len(m) > 0 else None
        
        if mergeTo == None:
            mergeTo = len(newClusters)
            newClusters.append([])
            
        '''
        print "Merting To:", mergeTo
        print "Next Cluster:", nextCluster
        print "New Clusters:", len(newClusters)
        '''
        
        for s in same:
            if (s in merged):
                continue
            
            merged[s] = mergeTo
            
            newClusters[mergeTo].append(s)
           
        for d in different:
            if (d in merged):
                continue
            
            merged[d] = len(newClusters)

            newClusters.append([d])
        
    
    result = []
    
    newClusters = filter(lambda c: len(c) > 0, newClusters)
    
    for c in newClusters:
        
        indecies = map(lambda  x: idx[x], c)
        _docs = reduce(lambda x,y: x+y, map(lambda x: clusters[x].members, indecies))
        
        cl = Cluster(_docs, c[0])
        result.append(cl)
    
    return result


def reclusterWithOPTICS(clustering, resolution=10, display=9):
    print "Aproximating Distances"
    distances = pseudoDistance(clustering)

    
    print "Running OPTICS"
    newClusters = []
    clusters = []
    centers = map(lambda x: x.center, clustering)
    for i,d in enumerate(distances):
    #d = distances[0]
        subclusters = []
        
        assert(d[0][0] == 0)
        
        label = clustering[i].label
        
        #map from reachability point to document
        revIdx = lambda x: clustering[i].members[x._id-1] if (x._id >0) else clustering[i].center 
        
        print
        print
        print "Cluster:", i
        #Output is a list of reachability points
        output = OPTICS(d,resolution, clustering[i])
        
        displayTruePlot(output, clustering[i], resolution, display)
        
        #Clustering of reachability points
        outputClusters = separateClusters(output, resolution)
        #remove the cluster center
        del outputClusters[0][0]

        #subclustering of documents
        subclusters = map(lambda c:map(revIdx, c) , outputClusters)
        
        '''
        if (len(subclusters) == 1):
            cl = clustering[i]
            cl.set_label()
            clusters.append(cl)
            continue
        '''
        #The largest minima
        minimum = selectMinimum(output,resolution)[:display-1]
        reps = map(revIdx, minimum)
        
        ''' 
        same, different = simulateUserFeedback(label, reps)
        
        
        print "Minima:", map(lambda x: (x.reachability, output.index(x)),minimum)
        print "Same:", len(same)
        print "Different:", len(different)
        
        #displayPoints([centers[i]] + reps)
        
        mergedClusters = []
        for rep in same:
            for cl in subclusters:
                if rep in cl:
                    mergedClusters += cl
                    subclusters.remove(cl)
        '''
        '''
        if (len(mergedClusters) > 0):
            subclusters = [mergedClusters] + subclusters
            
        '''
        
        subclusters = filter(lambda x: len(x) > 0, subclusters)
        
        
        for c, cl in enumerate(subclusters):
            center = cl[0]
            map(lambda x: center.aggregate(x), cl[1:])
            newCluster = Cluster(members=cl, center=center)
            newCluster.set_label()
            newClusters.append(newCluster)
        '''
        if(len(mergedClusters) > 0):
            cl = Cluster(members=mergedClusters, center=clustering[i].center)
            cl.set_label()
            clusters.append(cl)
        '''

        '''
        purity = validate([clustering[i]])
        print "Original Purity", purity
        print
        purities = validate(newClusters, display=True)
        print "Average Purity:", sum(purities)/len(purities)
        '''
            
        #raw_input()
        
    
    print "Merging Clusters"
    
    print "Number of Clusters before merge:", len(newClusters)
    clusters = mergeClusters(newClusters, display)
    return clusters


def main(args):

    if(len(args) != 2):
                print "Usage: mds.py clustering.pkl"
                print "     C is the cluster in clustering.pkl to display"           
                sys.exit(0)

    path = args[1]
                
    print "Loading"
    clustering = utils.load_obj(path)

    #map(lambda c: c.set_label(), clustering)

    clusters = reclusterWithOPTICS(clustering)
    
    _docs = reduce(lambda x,y: x+y, map(lambda c: c.members, clusters))
    
    
    confirm = BaseCONFIRM(_docs)
    confirm.clusters = clusters
    
    print "Original Number of Clusters:", len(clustering)
    print "Final Number of Clusters:", len(clusters)
    
    '''print reps
    
        imgs = []
        
        for idx in reps:
            if idx == 0:
                imgs.append(clustering[i].center)
            else:
                idx = idx -1
                imgs.append(clustering[i].members[idx])
                
        
        display(imgs)'''
    #print  len(selectWithHac(clustering))

    #print streamSelector(clustering)

    #print entropy(clustering)
    
    #print "Analyzing"
    analyzer = metric.KnownClusterAnalyzer(confirm)
    analyzer.print_all()

    print "User Queries:", QueryCount

if __name__ == '__main__':
    main(sys.argv) 




































