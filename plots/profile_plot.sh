
function plotProfile()
{
	title="$3"
	gnuplot << EOF
	set terminal jpeg size 1300, 600
	set output "output/$title.jpeg"

	set title "$title"
	set xlabel "Row/Column"
	set ylabel "Value sum"
	plot "$1" using 1:$2 with $4
EOF
}

commandline_args=("$@")
for arg in "${commandline_args[@]}"; do
	plotProfile "$arg" 2 "${arg%.plot}_orig" lines
	plotProfile "$arg" 3 "${arg%.plot}_orig_extrema" points
	plotProfile "$arg" 4 "${arg%.plot}_blurred" lines
	plotProfile "$arg" 5 "${arg%.plot}_blurred_extrema" points
done

