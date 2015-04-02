#!/bin/bash

num_threads=5
datasets="wales_balanced"
ks="12,24,36"
subsets="500,1000,2000,4000,4800"
seeds="30,50,75"
min_pts="10,20,30"
init_only="0"

t="1 2 3 4 5"

nice parallel --gnu -j $num_threads ./run.sh {1} {2} {3} {4} {5} {6} {7} ::: $datasets ::: $ks ::: $subsets ::: $seeds ::: $min_pts ::: $init_only ::: $t
