
DATE=$(date +%F)
out=${DATE}_feature_${1}_${2}.txt
python driver.py feature $1 > archive/$out 2> err/$out
