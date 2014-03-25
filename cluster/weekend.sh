
for size in small medium large
do
	./perfect.sh $size
	for eps in 0.4 0.3 0.2 0.1
	do
		./cluster.sh $size $eps
	done
done

 
