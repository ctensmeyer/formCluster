
num_threads=5
datasets="padeaths_all wales_balanced"
#datasets="wales_20"

t="1 2 3 4 5"
nice parallel --gnu -j $num_threads ./extract.sh {1} {2} ::: $datasets ::: $t

