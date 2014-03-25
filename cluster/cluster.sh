
DATE=$(date +%F)
out=${DATE}_cluster_${1}_${2}.txt
python driver.py cluster $1 $2 > archive/$out 2> err/$out
