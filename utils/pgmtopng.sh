#!/bin/bash

# converts all pgm files to png two directories below $1.  Then deletes the pgm files

function handle_dir {
	echo $1
	#mogrify -format png ephemeral:$1/*.pgm
	mogrify -format png $1/*.pgm
}
export -f handle_dir

indir=$1
subdirs=$1/*

parallel --gnu -j 5 handle_dir {1} ::: $subdirs

