#!/bin/bash

function dotypecluster
{
	dataset=$1
	num=$2
	path=~/formCluster/analysis/experiments/paper/init_kumar/datasets/
	python typecluster.py $path/$dataset/type_mats/$num $path/$dataset/data/type_results_${num}.txt 60 > $path/$dataset/logs/type_log_${num}.txt 
}

export -f dotypecluster

datasets="padeaths_all"
nums="1 2 3 4 5"

nice parallel --gnu -j 3 dotypecluster {1} {2} ::: $datasets ::: $nums

