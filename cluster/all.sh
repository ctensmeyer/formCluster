
DATE=$(date +%F)
out=${DATE}_all_${1}_${2}_${3}_${4}_${5}_${6}.txt
python driver.py all $1 $2 $3 $4 $5 $6 > archive/$out 2> err/$out
