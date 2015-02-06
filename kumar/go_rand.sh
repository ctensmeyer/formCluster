#!/bin/bash

function dorandcluster
{
	dataset=$1
	num=$2
	path=~/formCluster/analysis/experiments/paper/init_kumar/datasets/
	python randcluster.py $path/$dataset/rand_mats/$num $path/$dataset/data/rand_results_${num}.txt 60 > $path/$dataset/logs/rand_log_${num}.txt 
}

export -f dorandcluster

datasets="padeaths_all wales_balanced"
nums="1 2 3 4 5"

nice parallel --gnu -j 2 dorandcluster {1} {2} ::: $datasets ::: $nums

