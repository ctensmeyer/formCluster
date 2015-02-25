#!/bin/bash

num_threads=3
datasets="wales_balanced"

t="1 2 3"

nice parallel --gnu -j $num_threads ./feature.sh {1} {2} ::: $datasets ::: $t
