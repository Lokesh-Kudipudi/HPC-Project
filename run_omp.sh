#!/bin/zsh
for i in 8; do
  export OMP_NUM_THREADS=$i
  echo "Running with $i threads" >> output_omp_$i.txt
  echo -e "\n" >> output_omp_$i.txt
  for img in inputs/sample*.jpg; do
    echo "Processing $img" >> output_omp_$i.txt
    ./main_omp "$img" >> output_omp_$i.txt
    echo -e "\n\n" >> output_omp_$i.txt
  done
  echo -e "\n" >> output_omp_$i.txt
done