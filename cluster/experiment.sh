
datasets="wash_small wash_medium wash_big"
threshs="0.7"
models="base region weighted wavg"
parallel --gnu -j 6 ./cluster.sh {1} {2} {3} ::: $datasets ::: $threshs ::: $models


