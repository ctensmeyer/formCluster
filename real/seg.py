
import Image
import colorsys
import sys
import os
import random
import numpy as np
import math
import itertools


def log_add(logX, logY):
    if logY > logX:
        logX, logY = (logY, logX)
    if logX == float('-inf'):
        return logX

    diff = logY - logX
    #assert diff < 0
    if diff < -20:
        return logX

    return logX + math.log1p(math.exp(diff))
    
def log_sum_all(seq):
    return reduce(lambda x, y: log_add(x, y), seq, float('-inf'))
    #one = reduce(lambda x, y: log_add(x, y), seq, float('-inf'))
    #two = seq[0]
    #jfor val in seq[1:]:
    #j    two = log_add(two, val)
    #jprint "log sums: ", one, two
    #jreturn two
        

def rgb_to_hsv(pix):
    return colorsys.rgb_to_hsv(*pix)
    #return pix

def one_d_data_set():
    cluster = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    offset = 10
    points = [(x,) for x in cluster]
    points += [((x + offset),) for x in cluster]
    return points

def two_d_data_set():
    cluster1 = [ (1, 0), (2, 2), (3, -1), (2, 2), (2, 1), (2, 3)]
    cluster2 = [ (10, 15), (12, 14), (11, 14), (12, 12), (15, 10), (13, 13)]
    #offset = 20
    points = cluster1 + cluster2
    #points += [ (x[0] + offset, x[1] + offset) for x in points]
    return points

def listafy(im):
    pix = im.load() 
    l = []
    for x in xrange(im.size[0]):
        for y in xrange(im.size[1]):
            rgb = pix[x, y]
            hsi = rgb_to_hsv(rgb)
            #p = (x, y) + rgb
            p = (x, y) + hsi
            #p = rgb
            l.append(p)
    return l

def covar(vals1, mean1, vals2, mean2): 
    s = 0.0
    l = len(vals1)
    for x in xrange(l):
        s += (vals1[x] - mean1) * (vals2[x] - mean2)
    return s / l

def covar_mat(vals, means):
    mat = []
    for x in xrange(len(means)):
        row = []
        for y in xrange(len(means)):
            vals1 = map(lambda row: row[x], vals)
            vals2 = map(lambda row: row[y], vals)
            row.append(covar(vals1, means[x], vals2, means[y]))
        mat.append(row)
    return mat
    

def init_model(pixels, K):
    # mean
    means = []
    ranges = []
    for n in xrange(len(pixels[0])):
        values = map(lambda row: row[n], pixels)

        mean = sum(values) / float(len(values))
        means.append(mean)

        _range = (min(values), max(values))
        ranges.append(_range)
    means = tuple(means)
    #print means
    covars = covar_mat(pixels, means)
    #print covars
    std_devs = []
    for n in xrange(len(covars)):
        std_devs.append(math.sqrt(covars[n][n]))
    #print "std_devs", std_devs

    model = []
    for k in xrange(K):
        mean = perturb_means(means, std_devs)
        mat = perturb_mat(covars)
        #mat = np.diag(np.diag(mat))
        model.append( (mean, mat, 1.0 / K) )  # last num is alpha, cluster size
    return model
        

def perturb_means(means, std_devs):
    new_means = []
    for mean, std_dev in zip(means, std_devs):
        new_mean = mean + (random.random() - 0.5) * (0.75 * std_dev)
        new_means.append(new_mean)
    return np.array(new_means)
        

def perturb_mat(mat):
    all_vals = []
    for row in mat:
        for val in row:
            scale = random.random() * 0.2 + 0.4
            new_val = val  * scale
            all_vals.append(new_val)
    mat = np.array(all_vals).reshape(len(mat), len(mat))
    fix_mat(mat)
    return mat

def estimate(pixel, model):
    p = []
    _sum = 0.0
    for mean, covars, alpha in model:
        val = alpha * n_gauss(pixel, mean, covars)
        if val == 0:
            print "val is 0!!!! for point", pixel
        _sum += val
        p.append(val)
    p = map(lambda x: x / _sum, p)
    return p

def lprobs_to_probs(lprobs):
    mult = max(lprobs) - 10 
    scaled_probs = map(lambda x: math.exp(x - mult), lprobs) # scale probs to workable range
    _sum = sum(scaled_probs)
    probs = map(lambda x: x / _sum, scaled_probs)
    return probs
    
def estimate_log(pixel, model):
    ''' Do estimation in log space, but return normal probs '''
    lprobs = []
    for mean, covars, alpha in model:
        lprob = math.log(alpha) + log_n_gauss(pixel, mean, covars)
        lprobs.append(lprob)
    return lprobs

_small = 1e-10
def smooth_probs(probs):
    _sum = sum(probs)
    for x, prob in enumerate(probs):
        if prob < _small:
            probs[x] = prob + _small
            _sum += _small
    return [prob / _sum for prob in probs]
    

def E_step(pixels, model):
    # return assignements (cluster probabilities...)
    log_data_likelihood = 0.0
    assignments = []
    for pixel in pixels:
        lprobs = estimate_log(pixel, model)
        point_log_likelihood = log_sum_all(lprobs)
        #if point_log_likelihood > 0:
        #    print "\tthis is bad. PLL:", point_log_likelihood
        #    print "\tlprobs that did this:", lprobs
            
        log_data_likelihood += point_log_likelihood

        probs = lprobs_to_probs(lprobs)
        probs = smooth_probs(probs)
        assignments.append(probs)
        #print probs
    return assignments, log_data_likelihood

_epsilon = 1e-100
def fix_mat(mat):
    n = len(mat)
    for x in xrange(n):
        for y in xrange(n):
            if abs(mat[x][y]) < _epsilon:
                mat[x][y] = _epsilon * (1 + random.random())
    
def M_step(pixels, assignments):
    C = []
    N = len(pixels)
    K = len(assignments[0])
    D = len(pixels[0])
    for j in xrange(K):
        C.append(sum(map(lambda row: row[j], assignments)))
    print "\tC: ", C
    model = []
    for j in xrange(K):
        #print "Cluster:", j
        mean = np.zeros(D)
        #print "Mean Calc"
        for t in xrange(N):
            val = assignments[t][j] * np.array(pixels[t])
            #print "%s * %s = %s" % (assignments[t][j], np.array(pixels[t]), val)
            mean += val
        #print "before:", repr(mean)
        mean /= C[j]
        #print "mean:", repr(mean)

        #print "Mat Calc"
        mat = np.zeros((D, D))
        for t in xrange(N):
            pixel = np.array(pixels[t])
            #print "pix:", pixel
            diff = pixel - mean
            diff = diff.reshape(D, 1)
            trans = diff.transpose()
            val_mat = assignments[t][j] * diff * trans
            #print "%s * %s * %s = %s" % (assignments[t][j] , diff , trans, val_mat)
            mat += val_mat
        mat /= C[j]
        #mat = np.diag(np.diag(mat))
        fix_mat(mat)

        model.append((mean, mat, C[j] / N))
    return model
    # return model
    
def threshold_assignments(assignments):
    new_assigs = []
    for assignment in assignments:
        _max = max(assignment)
        new_assig = map(lambda x: 1.0 if x == _max else 0.0, assignment)
        new_assigs.append(new_assig)
    return new_assigs

def det(mat):
    return np.linalg.det(mat)

def invert(mat):
    return np.linalg.inv(mat)

_2pi = 2 * math.pi
def log_n_gauss(point, mean, covar):
    D = len(mean)
    const = -0.5 * D * math.log(_2pi)

    _det = abs(det(covar))
    if _det == 0:
        #print "_det is 0..."
        _det = 1e-100
    det_term = -0.5 * math.log(_det)

    # long part
    inverse = invert(covar)
    vector = point - mean
    vector = vector.reshape(D, 1)
    transpose = vector.transpose()
    prod1 = np.dot(transpose, inverse)
    prod2 = np.dot(prod1, vector)
    exp_term = -0.5 * prod2  # for exponent

    lprob = const + det_term + exp_term
    if lprob > 0:
        ''' this is okay, because is a probability mass...'''
        pass
        #print "\tlprob:", lprob
        #print "\tpixel:", point
        #print "\tmean:", mean
        #print "\tcovars:", covar
        #print "\tinverse:", inverse
        #print "\tnumpy det:", _det
        #print "\tmy det:", det_3(covar)
        #print "\tvector:", vector
        #print "\ttranspose:", transpose
        #print "\tprod1:", prod1
        #print "\tprod2:", prod2
        #print "\tconst:", const
        #print "\tdet_term:", det_term
        #print "\texp_term:", exp_term
        #exit()

        #print "\talpha:", alpha
    return lprob
    

def det_3(mat):
    _det = mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
    _det -=mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
    _det +=mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0])
    return _det

def n_gauss(point, mean, covar):
    D = len(mean)
    _det = abs(det(covar))
    if _det == 0:
        print "oh no, the determinate is 0! for:", covar
    denum = math.sqrt( (_2pi ** D) * _det)
    #denum = ( (_2pi ** (n / 2.0)) * math.sqrt(_det))
    if denum == 0:
        print "the denums gonna be 0, abort!"
        return 0
    const = 1.0 / denum
    inverse = invert(covar)
    vector = point - mean
    vector = vector.reshape(D, 1)
    #print vector
    transpose = vector.transpose()
    #print transpose
    #print inverse
    prod = np.dot(transpose, inverse)
    #print prod
    prod = np.dot(prod, vector)
    #print prod
    _exp = -0.5 * prod
    return const * math.exp(_exp)

def threshold_assignments(assignments):
    new_assignments = []
    for assignment in assignments:
        _max = max(assignment)
        new_assignments.append(map(lambda x: 1 if x == _max else 0, assignment))
    return new_assignments

MIN_ITERS = 7
PRECISION = 1
def has_converged(arr):
    if len(arr) < MIN_ITERS:
        return False
    return abs(arr[-1] - arr[-MIN_ITERS]) < PRECISION

def make_ari_table(X, Y):
    ''' This makes a confusion matrix when run with ground truth clusters '''
    mat = []
    for x in X:
        row = []
        for y in Y:
            overlap = len(x & y)
            row.append(overlap)
        mat.append(row)
    return mat

def bi_2(num):
    return num * (num -1) / 2
    
def ari(clusters1, clusters2):
    ''' Adjusted Rand Index '''
    K = len(clusters1)
    table = make_ari_table(clusters1, clusters2)
    A = []  # row sums
    B = []  # col sums
    N = sum(map(lambda cluster: len(cluster), clusters1))  # total pixels
    for k in xrange(K):
        A.append(sum(table[k]))
        B.append(sum(map(lambda row: row[k], table)))

    bi_n = bi_2(N)
    bi_a = sum(map(bi_2, A))
    bi_b = sum(map(bi_2, B))
    #index = sum(map(bi_2, [cell for row in table for cell in row]))
    index = sum(map(bi_2, itertools.chain(*table)))
    expected_index = (bi_a * bi_b) / bi_n
    max_index = 0.5 * (bi_a + bi_b) 

    numer = index - expected_index
    denum = max_index - expected_index
    return numer / denum
    
def make_assignments(assignments):
    ''' Takes a list of cluster probs and outputs a list of cluster assignments (single #) '''
    hard_assignments = []
    for assignment in assignments:
        _max = max(assignment)
        cluster_id = assignment.index(_max)
        hard_assignments.append(cluster_id)
    return hard_assignments

def make_clusters(pixels, assignments, K):
    ''' Takes the list of pixels, the output of make_assignments(), and K to make a list of clusters (sets) '''
    clusters = [set() for x in xrange(K)]
    for pixel, assignment in zip(pixels, assignments):
        clusters[assignment].add(pixel)
    return clusters

#Ground_Truth = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
def partition(pixels):
    ''' Takes the list of rgb pixels and makes the ground truth assignment for syn.png '''
    hard_assignments = []
    for pixel in pixels:
        rgb = tuple(pixel[2:])
        cluster = rgb.index(max(rgb))
        hard_assignments.append(cluster)
    return hard_assignments

def make_overlay(clusters, size):
    im = Image.new('RGB', size)
    pix = im.load()
    _len = len(OVERLAY_COLORS)
    for k, cluster in enumerate(clusters):
        for pixel in cluster:
            x = pixel[0]
            y = pixel[1]
            try:
                pix[x, y] = OVERLAY_COLORS[k]
            except:
                print k
    return im
    
    

OVERLAY_COLORS = [(128, 128, 128), (128, 0, 0), (0, 128, 0), (0, 0, 128),
                  (0, 128, 128), (128, 0, 128), (128, 128, 0), (255, 0, 0),
                  (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
                  (255, 255, 0)]
EVAL = False
MAX_ITERS = 200
def main(im_file, out_file, out_overlay, K):
    im = Image.open(im_file).convert('RGB')
    pixels = listafy(im)
    #o = [x for x in pixels if tuple(x[2:]) not in Ground_Truth]
    #print o
    #exit()
    #pixels = two_d_data_set()
    #print pixels
    #pixels = simple_data_set()
    model = init_model(pixels, K)

    #for mean, mat in model:
    #    point = mean - 1
    #    print point, mean, mat
    #    print n_gauss(point, mean, mat)
    #model[0][0][0] = 5
    #model[0][1][0] = 5
    #model[1][0][0] = 10
    #model[1][1][0] = 5
        
    log_data_likelihoods = []
    iters = 0
    for n in xrange(MAX_ITERS):
        iters += 1
        print "\nITERATION", n
        print "Current Model:", model
        print "Doing E Step:"
        assignments, log_data_likelihood = E_step(pixels, model)
        print "Log Data Likelihood:", log_data_likelihood

        log_data_likelihoods.append(log_data_likelihood)
        if has_converged(log_data_likelihoods):
            break
        # For Hard EM, use the thresholded assignments for the M Step
        thresh_assignments = threshold_assignments(assignments)
        assignments = map(smooth_probs, thresh_assignments)
        #for pixel, assignment in zip(pixels, assignments):
        #    print "\t", pixel, assignment
        print "Doing M Step"
        model = M_step(pixels, assignments)
    print "\n\nDone!"
    #for pixel, assignment, thresh in zip(pixels, assignments, thresh_assignments):
    #    print pixel, assignment, thresh

    clusters = make_clusters(pixels, make_assignments(assignments), K)
    overlay = make_overlay(clusters, im.size)
    overlay.save(out_overlay)

    blended = Image.blend(im, overlay, 0.5)
    blended.save(out_file)

    print "Num Iterations:", iters

    if EVAL:
        real_clusters = make_clusters(pixels, partition(pixels), K)
        _ari = ari(real_clusters, clusters)
        conf_mat = make_ari_table(real_clusters, clusters)
        print "ARI:", _ari
        print "Confusion Mat:"
        for row in conf_mat:
            print row


if __name__ == "__main__":
    if len(sys.argv) < 5:
        raise Exception("[in_image out_image out_overlay K]")
    im_file = sys.argv[1]
    out_file = sys.argv[2]
    out_overlay = sys.argv[3]
    K = int(sys.argv[4])
    main(im_file, out_file, out_overlay, K)


