
num_threads=5
#datasets="padeaths_balanced washpass padeaths_all wales_balanced wales_small"
datasets="nist wales_large"

params="100"
models="pipeline"
t="1 2 3 4 5"
nice parallel --gnu -j $num_threads ./cluster.sh {1} {2} {3} {4} 0 ::: $datasets ::: $params ::: $models ::: $t

