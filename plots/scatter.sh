
function plotScatter
{
	title="$1"
	gnuplot << EOF
	set terminal jpeg size 1300, 600
	set output "output/$title.jpeg"

	set title "$title"
	set xlabel "x"
	set ylabel "y"
	plot "$1" using 1:2 with points
EOF
}

plotScatter $1 

