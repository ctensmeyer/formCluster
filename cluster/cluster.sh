
DATE=$(date +%F)
out=${DATE}_cluster_${1}_${2}_${3}_${4}_${5}.txt
python driver.py cluster $1 $2 $3 ${4} ${5} > archive/$out 2> err/$out
