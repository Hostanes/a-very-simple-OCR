

#!/bin/bash

# uses nnlib-par.c instead of nnlib.c

if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_source_file.c>"
    exit 1
fi

input_file="$1"
base_name="${input_file%.*}"

output_file="bin/${base_name}.o"

echo "Compiling $input_file to $output_file..."
gcc "$input_file" lib/nnlib-omp.c lib/matrix-math.c -O3 -march=native -ffast-math -lm -o "$output_file" -g -fopenmp

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running program..."
    ./"$output_file" omp.nn
else
    echo "Compilation failed"
    exit 1
fi
