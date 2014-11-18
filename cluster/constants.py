

# do lazy loading of documents.  It's a good thing
LOAD_DOC_LAZY = True

# prefix/suffix matching for edit distance in text lines
PARTIAL_TEXT_MATCHES = False

# use weight decay when aggregating two document together - eventually phases out infrequent features
DECAY = True
# magnitude of the decay

LINE_DECAY = 1.0 / 10
TEXT_DECAY = 1.0 / 15
FINAL_PRUNE_DIV = 10.0

# used to determine grid line offset tolerance when matching
LINE_THRESH_MULT = 0.05
TEXT_THRESH_MULT = 0.15

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
USE_TEXT = False
USE_HORZ = True
USE_VERT = True
USE_SURF = True


