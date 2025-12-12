#!/bin/bash

# ==============================================
# MPI Testing Script for HPCC Cluster
# Run this ONLY on gateway.hpcc.vn
# ==============================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_MPI_HPCC_Cluster_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MPI TESTS - DISTRIBUTED MEMORY PLATFORM"
echo "HPCC Cluster (gateway.hpcc.vn)"
echo "=============================================="
echo "Date: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# System Information
{
    echo "=== CLUSTER SYSTEM INFORMATION ==="
    echo "Hostname: $(hostname)"
    echo "Date: $(date)"
    echo "User: $(whoami)"
    echo "CPU Cores available: $(nproc)"
    which mpirun && mpirun --version | head -2
    echo ""
} | tee "$OUTPUT_DIR/system_info.txt"

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# ==============================================
# Test MPI Naive
# ==============================================
echo ""
echo "=============================================="
echo "Testing: MPI Naive Matrix Multiplication"
echo "=============================================="

cd mpi-naive
make clean && make

if [ $? -eq 0 ]; then
    {
        echo "=== MPI Naive Results ==="
        echo "Platform: HPCC Cluster"
        echo "Date: $(date)"
        echo ""
        
        for size in 100 1000 10000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            # Test different process counts (N must be divisible)
            for procs in 1 2 4 5 8 10 20; do
                if [ $((size % procs)) -eq 0 ]; then
                    echo ""
                    echo "--- Processes: $procs ---"
                    
                    if [ $size -le 1000 ]; then
                        mpirun -np $procs ./mpi_program $size 1
                    else
                        mpirun -np $procs ./mpi_program $size 0
                    fi
                fi
            done
            echo ""
        done
    } 2>&1 | tee "../$OUTPUT_DIR/mpi_naive_results.txt"
else
    echo "Build failed for MPI Naive" | tee "../$OUTPUT_DIR/mpi_naive_results.txt"
fi

cd "$BASE_DIR"

# ==============================================
# Test MPI Strassen
# ==============================================
echo ""
echo "=============================================="
echo "Testing: MPI Strassen Matrix Multiplication"
echo "=============================================="

cd mpi-strassen
make clean && make

if [ $? -eq 0 ]; then
    {
        echo "=== MPI Strassen Results ==="
        echo "Platform: HPCC Cluster"
        echo "Date: $(date)"
        echo ""
        
        for size in 100 1000 10000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            # Strassen uses 7 sub-problems, so test with 1 and 7 processes
            for procs in 1 7; do
                echo ""
                echo "--- Processes: $procs ---"
                
                if [ $size -le 1000 ]; then
                    mpirun -np $procs ./mpi_program $size 1
                else
                    mpirun -np $procs ./mpi_program $size 0
                fi
            done
            echo ""
        done
    } 2>&1 | tee "../$OUTPUT_DIR/mpi_strassen_results.txt"
else
    echo "Build failed for MPI Strassen" | tee "../$OUTPUT_DIR/mpi_strassen_results.txt"
fi

cd "$BASE_DIR"

# ==============================================
# Test Hybrid MPI + OpenMP
# ==============================================
echo ""
echo "=============================================="
echo "Testing: Hybrid MPI+OpenMP Strassen"
echo "=============================================="

cd hybrid-strassen
make clean && make

if [ $? -eq 0 ]; then
    {
        echo "=== Hybrid MPI+OpenMP Results ==="
        echo "Platform: HPCC Cluster"
        echo "Date: $(date)"
        echo ""
        
        for size in 100 1000 10000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            # Test combinations of MPI processes and OpenMP threads
            for procs in 1 2 4 7; do
                for threads in 1 2 4 8; do
                    echo ""
                    echo "--- Processes: $procs, Threads: $threads ---"
                    
                    export OMP_NUM_THREADS=$threads
                    
                    if [ $size -le 1000 ]; then
                        mpirun -np $procs ./main $size 1 $threads 128
                    else
                        mpirun -np $procs ./main $size 0 $threads 128
                    fi
                done
            done
            echo ""
        done
    } 2>&1 | tee "../$OUTPUT_DIR/hybrid_strassen_results.txt"
else
    echo "Build failed for Hybrid Strassen" | tee "../$OUTPUT_DIR/hybrid_strassen_results.txt"
fi

cd "$BASE_DIR"

# ==============================================
# Generate Summary
# ==============================================
echo ""
echo "Generating summary..."

{
    echo "=== MPI BENCHMARK SUMMARY (HPCC Cluster) ==="
    echo "Date: $(date)"
    echo ""
    echo "--- MPI Naive ---"
    grep -E "(Matrix Size|Processes:|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/mpi_naive_results.txt" 2>/dev/null || echo "No results"
    echo ""
    echo "--- MPI Strassen ---"
    grep -E "(Matrix Size|Processes:|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/mpi_strassen_results.txt" 2>/dev/null || echo "No results"
    echo ""
    echo "--- Hybrid MPI+OpenMP ---"
    grep -E "(Matrix Size|Processes:|Threads:|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/hybrid_strassen_results.txt" 2>/dev/null || echo "No results"
} > "$OUTPUT_DIR/SUMMARY.txt"

echo ""
echo "=============================================="
echo "MPI TESTING COMPLETE!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"