
# Creates subsets of the data

prefix=england_
src_dir=../data/current/1911England

for N in 20 100 500 1000 5000
do
	out_dir=../data/subsets/$prefix${N}
	rm -rf $out_dir
	python create_test_data.py $src_dir $N $out_dir
done

