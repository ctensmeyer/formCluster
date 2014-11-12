
DATE=$(date +%F)
out=${DATE}_cluster_${1}_${2}_${3}.txt
python driver.py cluster $1 $2 $3 > archive/$out 2> err/$out
