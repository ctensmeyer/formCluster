#!/bin/bash

num_threads=5
datasets="nist"
subsets="100 200 500 1000 2000 4000 8000"

t="1 2 3 4 5"

nice parallel --gnu -j $num_threads ./subset2.sh {1} {2} {3} ::: $datasets ::: $subsets ::: $t
