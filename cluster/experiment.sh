#!/bin/bash

num_threads=7
datasets="washpass"
subsets=""
initial_clusters=""
num_seeds=""
min_pts=""

t="1 2 3 4 5 6 7"

nice parallel --gnu -j $num_threads ./all.sh {1} {2} ::: $datasets ::: $t
