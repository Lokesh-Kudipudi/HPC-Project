#!/bin/zsh
for img in inputs/sample*.jpg; do
  echo "Processing $img" >> output_seq.txt
  ./main_seq "$img" >> output_seq.txt
  echo -e "\n\n" >> output_seq.txt
done
