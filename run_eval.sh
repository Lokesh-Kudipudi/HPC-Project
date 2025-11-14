#!/bin/zsh

i=1
for img in inputs/sample*.jpg(Nn); do
  echo "Processing $img" >> eval.txt
  uv run evaluation.py --original "$img" --segmented "output/seq/sar_segmented_sample$i.png" >> eval.txt
  echo -e "\n\n" >> eval.txt
  i=$((i+1))
done
