#!/bin/bash

SRC_DIR="/home/mighty/repos/datasets/hah_esszimmer/images"
OUT_DIR="/home/mighty/repos/datasets/hah_esszimmer_small/images"
SCALE=50

# Create the output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Use parallel with mogrify to process images
cd "$SRC_DIR" || exit
parallel mogrify -path "$OUT_DIR" -resize $SCALE% -format jpg ::: *.jpg

echo "Processing completed. Resized images are in $OUT_DIR."