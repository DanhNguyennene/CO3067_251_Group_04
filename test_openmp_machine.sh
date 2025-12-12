#!/bin/bash

# ==============================================
# OpenMP Testing Script for Shared-Memory Platforms
# Run this on each group member's laptop/notebook
# ==============================================

MACHINE_NAME="${1:-Laptop_$(whoami)}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_OpenMP_${MACHINE_NAME}_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "OPENMP TESTS - SHARED MEMORY PLATFORM"
echo "=============================================="
echo "Machine: $MACHINE_NAME"
echo "Date: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# System Information
{
    echo "=== SYSTEM INFORMATION ==="
    echo "Hostname: $(hostname)"
    echo "Date: $(date)"
    echo "Kernel: $(uname -r)"
    echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^[ \t]*//')"
    echo "CPU Cores: $(nproc)"
    echo "Threads per core: $(lscpu | grep 'Thread(s) per core' | awk '{print $4}')"
    echo "Memory: $(grep MemTotal /proc/meminfo | awk '{printf "%.2f GB", $2/1024/1024}')"
    echo ""
} | tee "$OUTPUT_DIR/system_info.txt"

# Get base directory
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# ==============================================
# Test OpenMP Naive
# ==============================================
echo ""
echo "=============================================="
echo "Testing: OpenMP Naive Matrix Multiplication"
echo "=============================================="

cd openmp-naive
make clean && make

if [ $? -eq 0 ]; then
    {
        echo "=== OpenMP Naive Results ==="
        echo "Machine: $MACHINE_NAME"
        echo "Date: $(date)"
        echo ""
        
        for size in 100 1000 10000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            for threads in 1 2 4 8 16; do
                echo ""
                echo "--- Threads: $threads ---"
                
                if [ $size -le 1000 ]; then
                    ./main $size 1 $threads 128
                else
                    ./main $size 0 $threads 128
                fi
            done
            echo ""
        done
    } 2>&1 | tee "../$OUTPUT_DIR/openmp_naive_results.txt"
else
    echo "Build failed for OpenMP Naive" | tee "../$OUTPUT_DIR/openmp_naive_results.txt"
fi

cd "$BASE_DIR"

# ==============================================
# Test OpenMP Strassen
# ==============================================
echo ""
echo "=============================================="
echo "Testing: OpenMP Strassen Matrix Multiplication"
echo "=============================================="

cd openmp-strassen
make clean && make

if [ $? -eq 0 ]; then
    {
        echo "=== OpenMP Strassen Results ==="
        echo "Machine: $MACHINE_NAME"
        echo "Date: $(date)"
        echo ""
        
        for size in 100 1000 10000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            for threads in 1 2 4 8 16; do
                echo ""
                echo "--- Threads: $threads ---"
                
                if [ $size -le 1000 ]; then
                    ./optimized_main $size 1 $threads 128
                else
                    ./optimized_main $size 0 $threads 128
                fi
            done
            echo ""
        done
    } 2>&1 | tee "../$OUTPUT_DIR/openmp_strassen_results.txt"
else
    echo "Build failed for OpenMP Strassen" | tee "../$OUTPUT_DIR/openmp_strassen_results.txt"
fi

cd "$BASE_DIR"

# ==============================================
# Generate Summary
# ==============================================
echo ""
echo "Generating summary..."

{
    echo "=== OPENMP BENCHMARK SUMMARY ==="
    echo "Machine: $MACHINE_NAME"
    echo "Date: $(date)"
    echo ""
    echo "--- OpenMP Naive ---"
    grep -E "(Matrix Size|Threads:|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/openmp_naive_results.txt" 2>/dev/null || echo "No results"
    echo ""
    echo "--- OpenMP Strassen ---"
    grep -E "(Matrix Size|Threads:|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/openmp_strassen_results.txt" 2>/dev/null || echo "No results"
} > "$OUTPUT_DIR/SUMMARY.txt"

echo ""
echo "=============================================="
echo "OPENMP TESTING COMPLETE!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"
echo ""
echo "Run this script on each group member's laptop:"
echo "  Member 1: ./test_openmp_machine.sh Laptop1"
echo "  Member 2: ./test_openmp_machine.sh Laptop2"
echo "  Member 3: ./test_openmp_machine.sh Laptop3"