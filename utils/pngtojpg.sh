#!/bin/bash

# was used to convert the NIST tax forms images from png to jpg (but not the line images)

function handle_dir {
	echo $1
	mogrify -format jpg ephemeral:$1/*[0-9].png
}
export -f handle_dir

indir=$1
subdirs=$1/*

parallel --gnu -j 7 handle_dir {1} ::: $subdirs

