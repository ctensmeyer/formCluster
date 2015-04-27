#!/bin/bash

DATE=$(date +%F)
out=${DATE}_nosplit_${1}_${2}_${3}_${4}_${5}_${6}.txt
python driver.py run $1 $2 $3 $4 $5 > archive/$out 2> err/$out
