
#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_source_file.c>"
    exit 1
fi

input_file="$1"
base_name="${input_file%.*}"

output_file="${base_name}.o"

echo "Compiling $input_file to $output_file..."
gcc "$input_file" nnlib.c lib-omp/matrix-math.c -lm -o "$output_file" -fopenmp

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running program..."
    ./"$output_file"
else
    echo "Compilation failed"
    exit 1
fi
