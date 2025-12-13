#!/bin/bash

MACHINE_NAME="${1:-Machine_$(whoami)}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_OpenMP_${MACHINE_NAME}_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

{
    echo "Hostname: $(hostname)"
    echo "Date: $(date)"
    echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
    echo "Cores: $(nproc)"
} | tee "$OUTPUT_DIR/system_info.txt"

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

cd openmp-naive
make clean && make

if [ $? -eq 0 ]; then
    {
        echo "=== OpenMP Naive ==="
        for size in 100 1000 10000; do
            echo "Size: ${size}x${size}"
            for threads in 1 2 4 8 16; do
                echo "Threads: $threads"
                if [ $size -le 1000 ]; then
                    ./main $size 1 $threads 128
                else
                    ./main $size 0 $threads 128
                fi
            done
        done
        echo ""
        echo "=== Testing Recursive Blocked Method ==="
        for threads in 4 8; do
            echo "Threads: $threads, Block: 128"
            ./main 1000 0 $threads 128
        done
    } 2>&1 | tee "../$OUTPUT_DIR/openmp_naive_results.txt"
fi

cd "$BASE_DIR"

cd openmp-strassen
make clean && make

if [ $? -eq 0 ]; then
    {
        echo "=== OpenMP Strassen ==="
        for size in 100 1000 10000; do
            echo "Size: ${size}x${size}"
            for threads in 7 14 21 28; do
                echo "Threads: $threads"
                if [ $size -le 1000 ]; then
                    ./optimized_main $size 1 $threads 128
                else
                    ./optimized_main $size 0 $threads 128
                fi
            done
        done
    } 2>&1 | tee "../$OUTPUT_DIR/openmp_strassen_results.txt"
fi

cd "$BASE_DIR"

{
    echo "=== Summary ==="
    grep -E "(Size|Threads|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/openmp_naive_results.txt" 2>/dev/null
    grep -E "(Size|Threads|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/openmp_strassen_results.txt" 2>/dev/null
} > "$OUTPUT_DIR/SUMMARY.txt"

echo "Results: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"