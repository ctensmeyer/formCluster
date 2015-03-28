#!/bin/bash

DATE=$(date +%F)
out=${DATE}_subset_${1}_${2}_${3}.txt
python driver.py subset $1 $2 $3 > archive/$out 2> err/$out
