
num_threads=2
#datasets="washpass"
datasets="wales_20"

params="3"
models="pipeline"
t="1"
nice parallel --gnu -j $num_threads ./cluster.sh {1} {2} {3} {4} 0 ::: $datasets ::: $params ::: $models ::: $t

