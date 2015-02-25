
import math
import sys

# do lazy loading of documents.  It's a good thing
LOAD_DOC_LAZY = False

# prefix/suffix matching for edit distance in text lines
PARTIAL_TEXT_MATCHES = True

# use weight decay when aggregating two document together - eventually phases out infrequent features
DECAY = True

# magnitude of the decay
LINE_DECAY = 1.0 / 10
TEXT_DECAY = 1.0 / 15
FINAL_PRUNE_DIV = 10.0

# Push away constants
PUSH_AWAY_PERC = 0.01

# used to determine grid line offset tolerance when matching
LINE_THRESH_MULT = 0.05
TEXT_THRESH_MULT = 0.25

# determines the partiioning of the images in regions
REGION_ROWS = 4
REGION_COLS = 4

# For text line matching
TEXT_SIZE_RATIO = 0.8
TEXT_EDIT_DIST_THRESH = 0.2

# Draw settings
TEXT_COLOR = "black"
TEXT_COUNT_COLOR = "blue"
VERT_COLOR = "blue"
HORZ_COLOR = "red"
GRID_LINE_COUNT_COLOR = "black"

# Features Used
USE_TEXT = True
USE_HORZ = True
USE_VERT = True
USE_SURF = False

#USE_TEXT = False
#USE_HORZ = False
#USE_VERT = False
#USE_SURF = True

# New Clustering constants
NUM_TREES = 2000
FUNCTION_NUM_FEATURES = lambda x: int(math.sqrt(x))
RF_THREADS = 1
SIZE_OF_RANDOM_DATA = 1

REMOVE_DUP_FEATURES = False
DUP_THRESH = 0.01

FEATURES_OUTDIR="features"

descrip = sys.argv[2] if len(sys.argv) > 2 else ""

if descrip.startswith("wales_large"):
	NUM_CLUSTERS=[10,15,20,25,26,27,28,29,30,31,32,33,34,35,40,45,50,55,60]
elif descrip.startswith("wales_small"):
	NUM_CLUSTERS=[5,6,7,8,9,10,11,12,14,17,20]
elif descrip.startswith("wales_balanced"):
	NUM_CLUSTERS=[10,15,20,23,24,25,26,27,30,33,35,40]
elif descrip.startswith("washpass"):
	NUM_CLUSTERS=[2, 3, 4, 5, 6, 7]
elif descrip.startswith("nist"):
	NUM_CLUSTERS=[10,15,16,17,18,19,20,21,22,23,24,25,30,40]
elif descrip.startswith("padeaths_all"):
	NUM_CLUSTERS=[2, 3, 4, 5, 6, 7, 8, 9, 10]
elif descrip.startswith("padeaths_balanced"):
	NUM_CLUSTERS=[2, 3, 4, 5, 6, 7, 8, 9, 10]
elif descrip.startswith("wales_20"):
	#NUM_CLUSTERS=[2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 19]
	NUM_CLUSTERS=[2, 3, 4]
else:
	NUM_CLUSTERS=[2,5]
	

print "NUM_CLUSTERS", NUM_CLUSTERS

