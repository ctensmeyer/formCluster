#!/bin/bash

function dosamplecluster
{
	dataset=$1
	path=~/formCluster/analysis/experiments/paper/sampling_scheme/datasets/
	python samplecluster.py $path/$dataset/kumar_mats/$num $path/$dataset/data/kumar_results.txt 40 > $path/$dataset/logs/kumar_log.txt 
	#python samplecluster.py $path/$dataset/rand_mats/$num $path/$dataset/data/rand_results.txt 40 > $path/$dataset/logs/rand_log.txt 
	#python samplecluster.py $path/$dataset/type_mats/$num $path/$dataset/data/type_results.txt 40 > $path/$dataset/logs/type_log.txt 
}

export -f dosamplecluster

datasets="washpass padeaths_balanced padeaths_all wales_small wales_balanced"

nice parallel --gnu -j 5 dosamplecluster {1} ::: $datasets

