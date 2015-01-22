
set -e
function plot_compare
{
	metrics[2]='Acc'
	metrics[3]='V-measure'
	metrics[4]='Completeness'
	metrics[5]='Homogeneity'
	metrics[6]='ARI'
	metrics[7]='Silhouette'
	metric=${metrics[$1]}
	title="$metric"
	gnuplot << EOF
	set terminal jpeg
	set output "output/compare_${metric}.jpeg"

	set key outside
	set title "${metric} Compare"
	set xlabel "Perc Rand"
	set ylabel "${metric}"
	plot "$2" u 1:$1 with lines title "${2: -6:-4}", \
	     "$3" u 1:$1 with lines title "${3: -6:-4}", \
	     "$4" u 1:$1 with lines title "${4: -6:-4}", \
	     "$5" u 1:$1 with lines title "${5: -6:-4}", \
	     "$6" u 1:$1 with lines title "${6: -6:-4}", \
	     "$7" u 1:$1 with lines title "${7: -6:-4}", \
	     "$8" u 1:$1 with lines title "${8: -6:-4}", \
	     "$9" u 1:$1 with lines title "${9: -6:-4}", \
	     "${10}" u 1:$1 with lines title "${10: -6:-4}"
EOF
}

for x in 2 3 4 5 6 7
do
	plot_compare $x $1 $2 $3 $4 $5 $6 $7 $8 $9
done

