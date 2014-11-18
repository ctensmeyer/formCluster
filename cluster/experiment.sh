
#datasets="wales_100 wales_500 wales_1000 wales_twoclass_all wales_all"
datasets="wash_all"
threshs="0.50"
#models="perfect_base perfect_region perfect_weighted perfect_wavg base region weighted wavg best"
models="base region weighted wavg best"
parallel --gnu -j 5 ./cluster.sh {1} {2} {3} ::: $datasets ::: $threshs ::: $models

