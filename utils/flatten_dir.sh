#!/bin/bash

# this function will flatten the subdirectories of $1, that is, will copy all files out
# of subdir/Lines and subdir/OCR and put them in subdir/

function handle_dir() {
	echo $dir
	mv $1/Lines/* $1
	mv $1/OCR/* $1
	rmdir $1/Lines
	rmdir $1/OCR
}

for dir in $1/*
do
	handle_dir $dir
done

