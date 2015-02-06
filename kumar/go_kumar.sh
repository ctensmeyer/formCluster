#!/bin/bash

function doit
{
	python kcluster.py ~/formCluster/analysis/experiments/paper/init_kumar/datasets/$1/kumar_mats ~/formCluster/analysis/experiments/paper/init_kumar/datasets/$1/data/kumar_results.txt 60 > ~/formCluster/analysis/experiments/paper/init_kumar/datasets/$1/kumar_log.txt 
}

export -f doit

datasets="washpass padeaths_balanced padeaths wales_small wales_balanced nist"

parallel --gnu -j 2 doit {1} ::: $datasets

