#!/bin/bash

num_threads=7
datasets="wales_20"
subsets="5 10"
initial_clusters="2 3"
num_seeds="4"
min_pts="2"

t="1 2"

nice parallel --gnu -j $num_threads ./all.sh {1} {2} {3} {4} {5} {6} ::: $datasets ::: $subsets ::: $initial_clusters ::: $num_seeds ::: $min_pts ::: $t
