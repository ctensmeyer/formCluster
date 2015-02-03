
num_threads=6
datasets="washpass padeaths_balanced padeaths wales_balanced wales_small"
#datasets="wales_20"

t="1 2 3 4 5"
nice parallel --gnu -j $num_threads ./extract.sh {1} {2} ::: $datasets ::: $t

