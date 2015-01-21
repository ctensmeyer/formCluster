
num_threads=1
#datasets="wales_100 wales_500 wales_1000 wales_twoclass_all wales_all"
datasets="wales_all"

params="5 10 15 20 25 30 40 50 75 100"
params2="1 2 3 4"
#models="perfect_base perfect_region perfect_weighted perfect_wavg base region weighted wavg best"
models="pipeline"
parallel --gnu -j $num_threads ./cluster.sh {1} {2} {3} 20 0 ::: $datasets ::: $params ::: $models
parallel --gnu -j $num_threads ./cluster.sh {1} {2} {3} 20 1 ::: $datasets ::: $params2 ::: $models

