
num_threads=2
#datasets="washpass padeaths_balanced padeaths wales_balanced"
datasets="wales_20"

t="1 2"
nice parallel --gnu -j $num_threads ./extract.sh {1} {2} ::: $datasets ::: $t

