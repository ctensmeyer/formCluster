#!/bin/bash

num_threads=10
datasets="padeaths_balanced washpass padeaths_all"

t="1 2 3 4 5"

nice parallel --gnu -j $num_threads ./overall.sh {1} {2} ::: $datasets ::: $t
