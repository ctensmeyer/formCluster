#!/bin/bash

function dosamplecluster
{
	dataset=$1
	kind=$2
	path=~/formCluster/sampling_scheme/datasets/
	python samplecluster.py $path/$dataset/${kind}_mats $path/$dataset/data/${kind}_results.txt 40 > $path/$dataset/logs/${kind}_log.txt 
}

export -f dosamplecluster

datasets="washpass padeaths_balanced padeaths_all wales_small wales_balanced"
kinds="kumar rand type"

nice parallel --gnu -j 9 dosamplecluster {1} {2} ::: $datasets ::: $kinds

