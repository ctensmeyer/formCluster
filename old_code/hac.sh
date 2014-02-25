
rm -rf hac
mkdir hac
for (( c=$1; c>0; c--))
do
	mkdir hac/$c
	python cluster.py sim_mat.txt $c utils/otsu_output/ hac/$c
done

