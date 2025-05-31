#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <path/to/mnist-train-*.c>"
    exit 1
fi

input_file="$1"

# Validate file exists
if [ ! -f "$input_file" ]; then
    echo "Error: File '$input_file' does not exist."
    exit 1
fi

# Extract base name (e.g., mnist-train-batched)
base_name=$(basename "$input_file" .c)

# Extract directory (e.g., batched, serial, coalesced, omp)
dir_name=$(dirname "$input_file")

# Construct path to matching nnlib.c
nnlib_path="$dir_name/lib/nnlib.c"

if [ ! -f "$nnlib_path" ]; then
    echo "Error: Expected nnlib.c at '$nnlib_path' not found."
    exit 1
fi

# Output binary path
output_file="bin/${base_name}-unoptimized.out"

echo "================================================"
echo "Compiling WITHOUT optimizations:"
echo "- Input:        $input_file"
echo "- NN Lib:       $nnlib_path"
echo "- Output:       $output_file"
echo "- Optimization: None (no -O3, no -march=native, etc.)"
echo "- Threading:    No OpenMP thread binding"
echo "================================================"

echo "Compiling $input_file with $nnlib_path (no optimizations)..."
gcc "$input_file" "$nnlib_path" \
    -o "$output_file" \
    -lm -fopenmp --fast-math -O  # Only link math library (no other flags)

if [ $? -eq 0 ]; then
    echo "Compilation successful. Running unoptimized program..."
    echo "------------------------------------------------"
    ./"$output_file" model.nn
else
    echo "Compilation failed"
    exit 1
fi
