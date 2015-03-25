#!/bin/bash

DATE=$(date +%F)
out=${DATE}_overall_${1}_${2}.txt
python driver.py overall $1 $2 > archive/$out 2> err/$out
