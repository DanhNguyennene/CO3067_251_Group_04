#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="$HOME/CO3067_251_Group_04"
HOSTFILE="$BASE_DIR/hostfile"
OUTPUT_DIR="$BASE_DIR/results_MPI_HPCC_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

if [ "$1" == "--clean" ]; then
    if [ -f "$HOSTFILE" ]; then
        NODES=$(grep -oE 'MPI-node[0-9]+' "$HOSTFILE" | sort -u)
    else
        NODES="MPI-node1 MPI-node2 MPI-node3 MPI-node4 MPI-node5 MPI-node6 MPI-node7 MPI-node8 MPI-node9 MPI-node10 MPI-node11 MPI-node12"
    fi
    for node in $NODES; do
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "rm -rf ~/CO3067_251_Group_04" 2>/dev/null
    done
    rm -rf "$BASE_DIR"
    exit 0
fi

if [ ! -d "$BASE_DIR" ]; then
    cd "$HOME"
    git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git
fi

cd "$BASE_DIR"
git pull origin main 2>/dev/null || true

# Always regenerate hostfile with correct format for MPICH
cat > "$HOSTFILE" << 'EOF'
MPI-node1:4
MPI-node2:4
MPI-node3:4
MPI-node4:4
MPI-node6:4
MPI-node7:4
MPI-node8:4
MPI-node9:4
MPI-node10:4
MPI-node11:4
EOF

NODES=$(grep -oE 'MPI-node[0-9]+' "$HOSTFILE" | sort -u)
NODE_COUNT=$(echo "$NODES" | wc -w)

CURRENT_HOST=$(hostname)
REACHABLE_NODES=""

for node in $NODES; do
    if [ "$node" == "$CURRENT_HOST" ]; then
        REACHABLE_NODES="$node"
        break
    fi
done

for node in $NODES; do
    if [ "$node" != "$CURRENT_HOST" ]; then
        if ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "echo OK" 2>/dev/null >/dev/null; then
            REACHABLE_NODES="$REACHABLE_NODES $node"
        fi
    fi
done

REACHABLE_COUNT=$(echo $REACHABLE_NODES | wc -w)
if [ "$REACHABLE_COUNT" -eq 0 ]; then
    # No nodes reachable at all - shouldn't happen but fallback to localhost
    USE_HOSTFILE=false
    cat > "$HOSTFILE" << 'EOF'
localhost:4
EOF
    MPI_OPTS="--hostfile $HOSTFILE"
    echo "Warning: No nodes reachable. Running on localhost only."
elif [ "$REACHABLE_COUNT" -eq 1 ] && [ "$REACHABLE_NODES" == "$CURRENT_HOST" ]; then
    # Only current node, no other nodes reachable
    USE_HOSTFILE=false
    echo "Warning: Running on single node ($CURRENT_HOST) only."
    MPI_OPTS="--hostfile $HOSTFILE"
else
    # Multiple nodes reachable
    USE_HOSTFILE=true
    MPI_OPTS="--hostfile $HOSTFILE"
    echo "Running on $REACHABLE_COUNT nodes: $REACHABLE_NODES"
fi

if [ "$USE_HOSTFILE" = true ]; then
    for node in $REACHABLE_NODES; do
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" "
            if [ ! -d ~/CO3067_251_Group_04 ]; then
                cd \$HOME && git clone https://github.com/DanhNguyennene/CO3067_251_Group_04.git 2>/dev/null
            else
                cd ~/CO3067_251_Group_04 && git pull origin main 2>/dev/null
            fi
        " 2>/dev/null
    done
fi

{
    echo "Hostname: $(hostname)"
    echo "Date: $(date)"
    echo "Nodes: $NODE_COUNT (Reachable: $REACHABLE_COUNT)"
    mpirun --version 2>&1 | head -1
} | tee "$OUTPUT_DIR/system_info.txt"

compile_on_all_nodes() {
    local dir=$1
    if [ "$USE_HOSTFILE" = true ]; then
        for node in $REACHABLE_NODES; do
            ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" "cd ~/CO3067_251_Group_04/$dir && make clean && make" 2>&1 >/dev/null &
        done
        wait
    fi
}

cd "$BASE_DIR/mpi-naive"
make clean && make
compile_on_all_nodes "mpi-naive"

if [ -f ./mpi_program ]; then
    {
        echo "=== MPI Naive ==="
        # Small matrix with verification to check correctness
        echo "Size: 1000x1000 (with verification)"
        for procs in 10 20 40; do
            echo "Procs: $procs"
            mpirun $MPI_OPTS -np $procs ./mpi_program 1000 1
        done
        
        echo ""
        echo "=== Large matrices without verification ==="
        # Large matrices, no verification for speed
        for size in 5000 10000; do
            echo "Size: ${size}x${size}"
            for procs in 10 20 40; do
                if [ $((size % procs)) -eq 0 ]; then
                    echo "Procs: $procs"
                    mpirun $MPI_OPTS -np $procs ./mpi_program $size 0
                fi
            done
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_naive_results.txt"
fi

cd "$BASE_DIR/mpi-strassen"
make clean && make
compile_on_all_nodes "mpi-strassen"

if [ -f ./mpi_program ]; then
    {
        echo "=== MPI Strassen ==="
        # Small with verification
        echo "Size: 1024x1024 (with verification)"
        echo "Procs: 7"
        mpirun $MPI_OPTS -np 7 ./mpi_program 1024 1
        
        echo ""
        echo "=== Large matrices without verification ==="
        # Strassen requires 7 processes
        for size in 4096 8192; do
            echo "Size: ${size}x${size}"
            echo "Procs: 7"
            mpirun $MPI_OPTS -np 7 ./mpi_program $size 0
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_strassen_results.txt"
fi

cd "$BASE_DIR/hybrid-strassen"
make clean && make
compile_on_all_nodes "hybrid-strassen"

if [ -f ./main ]; then
    {
        echo "=== Hybrid MPI+OpenMP ==="
        # Small with verification
        echo "Size: 2048x2048 (with verification)"
        for procs in 10 20; do
            for threads in 2 4; do
                echo "Procs: $procs, Threads: $threads (Total: $((procs*threads)) workers)"
                export OMP_NUM_THREADS=$threads
                mpirun $MPI_OPTS -np $procs -x OMP_NUM_THREADS=$threads ./main 2048 1 $threads 128
            done
        done
        
        echo ""
        echo "=== Large matrices without verification ==="
        # Large matrices with high process/thread counts
        for size in 8192 10240; do
            echo "Size: ${size}x${size}"
            for procs in 10 20; do
                for threads in 2 4; do
                    echo "Procs: $procs, Threads: $threads (Total: $((procs*threads)) workers)"
                    export OMP_NUM_THREADS=$threads
                    mpirun $MPI_OPTS -np $procs -x OMP_NUM_THREADS=$threads ./main $size 0 $threads 128
                done
            done
        done
    } 2>&1 | tee "$OUTPUT_DIR/hybrid_strassen_results.txt"
fi

{
    echo "=== Summary ==="
    grep -E "(Size|Procs|Threads|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/mpi_naive_results.txt" 2>/dev/null
    grep -E "(Size|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/mpi_strassen_results.txt" 2>/dev/null
    grep -E "(Size|Procs|Threads|Total execution time|PASSED|FAILED)" "$OUTPUT_DIR/hybrid_strassen_results.txt" 2>/dev/null
} > "$OUTPUT_DIR/SUMMARY.txt"

echo "Results: $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"