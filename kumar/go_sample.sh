#!/bin/bash

function dosamplecluster
{
	dataset=$1
	path=~/formCluster/analysis/experiments/explore/sampling_scheme/datasets/
	python samplecluster.py $path/$dataset/kumar_mats/$num $path/$dataset/data/kumar_results2.txt 40 > $path/$dataset/logs/kumar_log2.txt 
	#python samplecluster.py $path/$dataset/rand_mats/$num $path/$dataset/data/rand_results.txt 40 > $path/$dataset/logs/rand_log.txt 
	#python samplecluster.py $path/$dataset/type_mats/$num $path/$dataset/data/type_results.txt 40 > $path/$dataset/logs/type_log.txt 
}

export -f dosamplecluster

datasets="wales_small"

nice parallel --gnu -j 5 dosamplecluster {1} ::: $datasets

