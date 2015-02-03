#!/bin/bash

num_threads=4
datasets="washpass padeaths_balanced padeaths wales_balanced"

params="50"
models="pipeline"
t="1 2"

nice parallel --gnu -j $num_threads ./cluster {1} {2} {3} {4} 0 ::: $datasets ::: $params ::: $models ::: $t
