#!/bin/bash

echo "compiling the OpenMP version"
./compile-and-run-omp.sh mnist-train.c


echo "compiling the OpenCL version"
./compile-and-run-ocl.sh mnist-train-ocl.c

echo "compiling the serial version"
./compile-and-run.sh mnist-train.c

./compile-and-run.sh mnist-test-models.c
