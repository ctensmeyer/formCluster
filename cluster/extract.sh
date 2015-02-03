
DATE=$(date +%F)
out=${DATE}_extract_${1}_${2}.txt
python driver.py extract $1 $2 > archive/$out 2> err/$out
