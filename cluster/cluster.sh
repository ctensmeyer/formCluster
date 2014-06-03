
DATE=$(date +%F)
out=${DATE}_cluster_${1}_${2}.txt
python driver.py twice $1 $2 > archive/$out 2> err/$out
