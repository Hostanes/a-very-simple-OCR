
#!/bin/bash

# Script to compile, run, and test both serial and parallel MNIST neural network implementations

# Configuration
INPUT_FILE="mnist-train.c"
MODEL_TESTER="mnist-test-models.c"
SERIAL_OUTPUT="mnist-test-serial.o"
PARALLEL_OUTPUT="mnist-test-parallel.o"
MODEL_TESTER_OUTPUT="mnist-model-tester"
SERIAL_MODEL="serial.nn"
PARALLEL_MODEL="parallel.nn"

CFLAGS="-O3 -march=native -ffast-math -lm -g -fopenmp"

compile_and_run() {
    local source=$1
    local output=$2
    local nnlib=$3
    local model=$4
    
    echo "Compiling $source with $nnlib..."
    gcc "$source" "$nnlib" lib/matrix-math.c $CFLAGS -o "$output"
    
    if [ $? -ne 0 ]; then
        echo "Compilation failed for $output"
        exit 1
    fi
    
    echo "Running $output to generate $model..."
    ./"$output" "$model"
    
    if [ $? -ne 0 ]; then
        echo "Execution failed for $output"
        exit 1
    fi
    
    echo "$model generated successfully"
}

compile_and_run "$INPUT_FILE" "$SERIAL_OUTPUT" "lib/nnlib.c" "$SERIAL_MODEL"

compile_and_run "$INPUT_FILE" "$PARALLEL_OUTPUT" "lib/nnlib-par.c" "$PARALLEL_MODEL"

echo "Compiling model tester..."
gcc "$MODEL_TESTER" lib/nnlib.c lib/matrix-math.c $CFLAGS -o "$MODEL_TESTER_OUTPUT"

if [ $? -ne 0 ]; then
    echo "Compilation failed for model tester"
    exit 1
fi

echo "Testing models..."
./"$MODEL_TESTER_OUTPUT" "$SERIAL_MODEL" "$PARALLEL_MODEL"

if [ $? -ne 0 ]; then
    echo "Model testing failed"
    exit 1
fi

echo "All tasks completed successfully"
