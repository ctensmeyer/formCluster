#!/bin/bash

num_threads=7
datasets="wales_20 wales_100"

t="1 2"

nice parallel --gnu -j $num_threads ./overall.sh {1} {2} ::: $datasets ::: $t
