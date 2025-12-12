#!/bin/bash

# ==============================================
# MPI Testing Script for 2-Machine Cluster
# Machines connected via WireGuard tunnel
# Machine 1: danhbuonba@10.0.0.2 (local)
# Machine 2: danhvuive@10.0.0.1 (remote)
# ==============================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results_MPI_Cluster_${TIMESTAMP}"

# Use common path that exists on both machines
BASE_DIR="$HOME/mpi_cluster_para_assignment"
HOSTFILE="$BASE_DIR/hostfile"

# MPI options for cluster (use WireGuard network 10.0.0.0/24)
MPI_OPTS="--hostfile $HOSTFILE --mca btl tcp,self --mca btl_tcp_if_include 10.0.0.0/24 --mca oob_tcp_if_include 10.0.0.0/24"

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "MPI TESTS - 2-MACHINE CLUSTER (WireGuard)"
echo "=============================================="
echo "Date: $(date)"
echo "Output: $OUTPUT_DIR"
echo "Base Dir: $BASE_DIR"
echo "=============================================="

# Create hostfile
cat > "$HOSTFILE" << 'EOF'
10.0.0.2 slots=8
10.0.0.1 slots=8
EOF

echo "Hostfile created:"
cat "$HOSTFILE"
echo ""

# Test cluster connectivity
echo "Testing cluster connectivity..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 10.0.0.1 "echo 'Remote OK'" 2>/dev/null; then
    echo "ERROR: Cannot connect to remote machine 10.0.0.1"
    echo "Make sure WireGuard is up and SSH is configured"
    exit 1
fi
echo "Cluster connectivity: OK"
echo ""

# Sync code to remote machine
echo "Syncing code to remote machine..."
rsync -az --exclude='*.o' --exclude='main' --exclude='mpi_program' --exclude='optimized_main' \
    "$BASE_DIR/" 10.0.0.1:"$BASE_DIR/"
echo "Sync complete"
echo ""

# System Information
{
    echo "=== CLUSTER SYSTEM INFORMATION ==="
    echo "Local Hostname: $(hostname)"
    echo "Remote Hostname: $(ssh 10.0.0.1 hostname)"
    echo "Date: $(date)"
    echo "User: $(whoami)"
    echo "Local CPU Cores: $(nproc)"
    echo "Remote CPU Cores: $(ssh 10.0.0.1 nproc)"
    echo "MPI Version:"
    mpirun --version | head -2
    echo ""
} | tee "$OUTPUT_DIR/system_info.txt"

cd "$BASE_DIR"

# ==============================================
# Test MPI Naive
# ==============================================
echo ""
echo "=============================================="
echo "Testing: MPI Naive Matrix Multiplication"
echo "=============================================="

cd "$BASE_DIR/mpi-naive"
make clean && make

# Compile on remote too
ssh 10.0.0.1 "cd $BASE_DIR/mpi-naive && make clean && make" 2>&1 | tail -3

if [ -f ./mpi_program ]; then
    {
        echo "=== MPI Naive Results ==="
        echo "Platform: 2-Machine Cluster (WireGuard)"
        echo "Date: $(date)"
        echo ""
        
        for size in 100 1000 4000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            # Test different process counts
            for procs in 1 2 4 8 16; do
                if [ $((size % procs)) -eq 0 ]; then
                    echo ""
                    echo "--- Processes: $procs ---"
                    
                    if [ $size -le 1000 ]; then
                        mpirun $MPI_OPTS -np $procs ./mpi_program $size 1
                    else
                        mpirun $MPI_OPTS -np $procs ./mpi_program $size 0
                    fi
                fi
            done
            echo ""
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_naive_results.txt"
else
    echo "Build failed for MPI Naive" | tee "$OUTPUT_DIR/mpi_naive_results.txt"
fi

# ==============================================
# Test MPI Strassen
# ==============================================
echo ""
echo "=============================================="
echo "Testing: MPI Strassen Matrix Multiplication"
echo "=============================================="

cd "$BASE_DIR/mpi-strassen"
make clean && make

# Compile on remote too
ssh 10.0.0.1 "cd $BASE_DIR/mpi-strassen && make clean && make" 2>&1 | tail -3

if [ -f ./mpi_program ]; then
    {
        echo "=== MPI Strassen Results ==="
        echo "Platform: 2-Machine Cluster (WireGuard)"
        echo "Date: $(date)"
        echo ""
        
        # MPI Strassen requires exactly 7 processes (one for each of the 7 Strassen sub-problems)
        for size in 100 1000 4000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            echo ""
            echo "--- Processes: 7 (required by Strassen algorithm) ---"
            
            if [ $size -le 1000 ]; then
                mpirun $MPI_OPTS -np 7 ./mpi_program $size 1
            else
                mpirun $MPI_OPTS -np 7 ./mpi_program $size 0
            fi
            echo ""
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_strassen_results.txt"
else
    echo "Build failed for MPI Strassen" | tee "$OUTPUT_DIR/mpi_strassen_results.txt"
fi

# ==============================================
# Test Hybrid MPI + OpenMP
# ==============================================
echo ""
echo "=============================================="
echo "Testing: Hybrid MPI+OpenMP Strassen"
echo "=============================================="

cd "$BASE_DIR/hybrid-strassen"
make clean && make

# Compile on remote too
ssh 10.0.0.1 "cd $BASE_DIR/hybrid-strassen && make clean && make" 2>&1 | tail -3

if [ -f ./main ]; then
    {
        echo "=== Hybrid MPI+OpenMP Results ==="
        echo "Platform: 2-Machine Cluster (WireGuard)"
        echo "Date: $(date)"
        echo ""
        
        # Hybrid Strassen requires exactly 7 MPI processes, but can vary OpenMP threads
        for size in 100 1000 4000; do
            echo "=========================================="
            echo "Matrix Size: ${size}x${size}"
            echo "=========================================="
            
            # Test with 7 MPI processes and different OpenMP thread counts
            for threads in 1 2 4 8; do
                echo ""
                echo "--- Processes: 7, Threads: $threads ---"
                
                if [ $size -le 1000 ]; then
                    mpirun $MPI_OPTS -np 7 -x OMP_NUM_THREADS=$threads ./main $size 1 $threads 128
                else
                    mpirun $MPI_OPTS -np 7 -x OMP_NUM_THREADS=$threads ./main $size 0 $threads 128
                fi
            done
            echo ""
        done
    } 2>&1 | tee "$OUTPUT_DIR/hybrid_strassen_results.txt"
else
    echo "Build failed for Hybrid Strassen" | tee "$OUTPUT_DIR/hybrid_strassen_results.txt"
fi

# ==============================================
# Generate Summary
# ==============================================
echo ""
echo "Generating summary..."

{
    echo "=== MPI BENCHMARK SUMMARY (2-Machine Cluster) ==="
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

# Copy results back to original directory
cp -r "$OUTPUT_DIR" /mnt/c/Just_Learning/University/sem_9/parallel/btl/Parallel_Computing/ 2>/dev/null || true

echo ""
echo "=============================================="
echo "MPI CLUSTER TESTING COMPLETE!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"
