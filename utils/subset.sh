
# Creates subsets of the data

prefix=padeaths_
src_dir=../data/current/PADeaths

for N in 20 100 500 1000 2000
do
	out_dir=../data/subsets/$prefix${N}
	rm -rf $out_dir
	python create_test_data.py $src_dir $N $out_dir
done

