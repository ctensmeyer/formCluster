
mkdir -p big_results
for ep in 0.15 0.2 0.25
do
	outfile=big_results/out_${ep}.txt
	#echo $outfile
	python driver.py $ep > $outfile &
done

