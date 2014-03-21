
DATE=$(date +%F)
python driver.py cluster $1 > archive/${DATE}_cluster_${1}.txt
