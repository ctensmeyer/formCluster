
prefix=wales
src_dir=../data/lines/1911Wales

for N in 40 100 1000
do
	out_dir=../data/$prefix${N}
	rm -rf $out_dir
	python create_test_data.py $src_dir $N $out_dir
done

