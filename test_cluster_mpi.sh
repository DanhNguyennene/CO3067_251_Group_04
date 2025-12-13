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
        NODES="MPI-node1 MPI-node2 MPI-node3"
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

if [ ! -f "$HOSTFILE" ]; then
    cat > "$HOSTFILE" << 'EOF'
MPI-node1 slots=4
MPI-node2 slots=4
MPI-node3 slots=4
EOF
fi

NODES=$(grep -oE 'MPI-node[0-9]+' "$HOSTFILE" | sort -u)
NODE_COUNT=$(echo "$NODES" | wc -w)

REACHABLE_NODES=""
for node in $NODES; do
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$node" "echo OK" 2>/dev/null >/dev/null; then
        REACHABLE_NODES="$REACHABLE_NODES $node"
    fi
done

REACHABLE_COUNT=$(echo $REACHABLE_NODES | wc -w)
if [ "$REACHABLE_COUNT" -eq 0 ]; then
    USE_HOSTFILE=false
    MPI_OPTS="--oversubscribe"
else
    USE_HOSTFILE=true
    MPI_OPTS="--hostfile $HOSTFILE --mca btl tcp,self"
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
        for size in 100 1000 4000; do
            echo "Size: ${size}x${size}"
            for procs in 1 2 4 8; do
                if [ $((size % procs)) -eq 0 ]; then
                    echo "Procs: $procs"
                    if [ $size -le 1000 ]; then
                        mpirun $MPI_OPTS -np $procs ./mpi_program $size 1
                    else
                        mpirun $MPI_OPTS -np $procs ./mpi_program $size 0
                    fi
                fi
            done
        done
        echo ""
        echo "=== Testing Pipelined Ring Method ==="
        for procs in 2 4 8; do
            echo "Procs: $procs"
            mpirun $MPI_OPTS -np $procs ./mpi_program 1000 0
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_naive_results.txt"
fi

cd "$BASE_DIR/mpi-strassen"
make clean && make
compile_on_all_nodes "mpi-strassen"

if [ -f ./mpi_program ]; then
    {
        echo "=== MPI Strassen ==="
        for size in 100 1000 4000; do
            echo "Size: ${size}x${size}"
            if [ $size -le 1000 ]; then
                mpirun $MPI_OPTS -np 7 ./mpi_program $size 1
            else
                mpirun $MPI_OPTS -np 7 ./mpi_program $size 0
            fi
        done
    } 2>&1 | tee "$OUTPUT_DIR/mpi_strassen_results.txt"
fi

cd "$BASE_DIR/hybrid-strassen"
make clean && make
compile_on_all_nodes "hybrid-strassen"

if [ -f ./main ]; then
    {
        echo "=== Hybrid MPI+OpenMP ==="
        for size in 100 1000 4000; do
            echo "Size: ${size}x${size}"
            for threads in 1 2 4 7; do
                echo "Procs: 7, Threads: $threads"
                export OMP_NUM_THREADS=$threads
                if [ $size -le 1000 ]; then
                    mpirun $MPI_OPTS -np 7 -x OMP_NUM_THREADS=$threads ./main $size 1 $threads 128
                else
                    mpirun $MPI_OPTS -np 7 -x OMP_NUM_THREADS=$threads ./main $size 0 $threads 128
                fi
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