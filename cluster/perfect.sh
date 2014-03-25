
DATE=$(date +%F)
out=${DATE}_perfect_${1}.txt
python driver.py perfect $1 > archive/$out 2> err/$out

