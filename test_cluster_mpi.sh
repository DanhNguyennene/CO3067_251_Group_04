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
# Intel Xeon E5-2680 v3: 12 cores, 24 threads per node
cat > "$HOSTFILE" << 'EOF'
MPI-node1:24
MPI-node2:24
MPI-node3:24
MPI-node4:24
MPI-node6:24
MPI-node7:24
MPI-node8:24
MPI-node9:24
MPI-node10:24
MPI-node11:24
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
        echo "Note: Matrix size must be divisible by number of processes"
        echo ""
        
        echo "Small matrix tests with varying process counts:"
        echo "Size: 960x960 (divisible by 4,8,16,24) - with verification"
        for procs in 4 8 16 24; do
            echo "Procs: $procs"
            mpirun $MPI_OPTS -np $procs ./mpi_program 960 1
        done
        
        echo ""
        echo "Medium matrix tests (safe sizes):"
        echo "Size: 3072x3072 (8 procs)"
        mpirun $MPI_OPTS -np 8 ./mpi_program 3072 0
        
        echo ""
        echo "Size: 5000x5000 (4 and 8 procs)"
        for procs in 4 8; do
            echo "Procs: $procs"
            mpirun $MPI_OPTS -np $procs ./mpi_program 5000 0
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_naive_results.txt"
fi

cd "$BASE_DIR/mpi-strassen"
make clean && make
compile_on_all_nodes "mpi-strassen"

if [ -f ./mpi_program ]; then
    {
        echo "=== MPI Strassen ==="
        echo "Note: MPI Strassen requires exactly 7 processes"
        echo ""
        
        echo "Small matrix (with verification):"
        echo "Size: 1024x1024, Procs: 7"
        mpirun $MPI_OPTS -np 7 ./mpi_program 1024 1
        
        echo ""
        echo "Medium matrix:"
        echo "Size: 2048x2048, Procs: 7"
        mpirun $MPI_OPTS -np 7 ./mpi_program 2048 0
        
        echo ""
        echo "Large matrix:"
        echo "Size: 4096x4096, Procs: 7"
        mpirun $MPI_OPTS -np 7 ./mpi_program 4096 0
        
        echo ""
        echo "Extra large matrix (safe size):"
        echo "Size: 5120x5120, Procs: 7"
        mpirun $MPI_OPTS -np 7 ./mpi_program 5120 0
    } 2>&1 | tee "$OUTPUT_DIR/mpi_strassen_results.txt"
fi

cd "$BASE_DIR/hybrid-strassen"
make clean && make
compile_on_all_nodes "hybrid-strassen"

if [ -f ./main ]; then
    {
        echo "=== Hybrid MPI+OpenMP ==="
        echo "Note: Hybrid uses 7 MPI processes with OpenMP threads per process"
        echo ""
        
        echo "Size: 2048x2048, MPI Procs: 7, OpenMP Threads: 3 - with verification"
        mpirun $MPI_OPTS -np 7 -genv OMP_NUM_THREADS 3 ./main 2048 1 3 128
        
        echo ""
        echo "Size: 3072x3072, MPI Procs: 7, OpenMP Threads: 8"
        mpirun $MPI_OPTS -np 7 -genv OMP_NUM_THREADS 8 ./main 3072 0 8 128
        
        echo ""
        echo "Size: 4096x4096, MPI Procs: 7, OpenMP Threads: 12"
        mpirun $MPI_OPTS -np 7 -genv OMP_NUM_THREADS 12 ./main 4096 0 12 128
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
