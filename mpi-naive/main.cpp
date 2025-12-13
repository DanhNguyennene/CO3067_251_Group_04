#include "mpi-naive.h"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <N> [verify]\n";
            std::cerr << "  N: Matrix size (must be divisible by number of processes)\n";
            std::cerr << "  verify: 0=skip, 1=verify (default: 0)\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);
    int verify = (argc > 2) ? std::atoi(argv[2]) : 0;

    std::vector<int> A, B, C;
    
    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N, 0);
        for (int i = 0; i < N * N; i++) {
            A[i] = 1 + std::rand() % 9;
            B[i] = 1 + std::rand() % 9;
        }
    } else {
        C.resize(N * N, 0);
    }

    double comp_time;
    double start_time = MPI_Wtime();
    
    pipelinedRingMultiply(N, rank, num_procs, A, B, C, comp_time);
    
    double total_time = MPI_Wtime() - start_time;

    if (rank == 0)
    {
        std::cout << "\nTotal execution time: " << total_time << " seconds\n";
        std::cout << "Computation time: " << comp_time << " seconds\n";
    }

    if (verify && rank == 0)
    {
        std::cout << "\nVerifying Correctness...\n";

        double verify_start = MPI_Wtime();
        std::vector<int> C_verify(N * N, 0);
        serialVerify(N, A, B, C_verify);
        double verify_time = MPI_Wtime() - verify_start;

        std::cout << "Serial verification time: " << verify_time << "s\n";

        bool passed = verifyResults(N, C, C_verify, rank);

        if (passed)
        {
            std::cout << "\nSpeedup vs Serial: " << std::fixed << std::setprecision(2)
                      << verify_time / total_time << "x\n";
        }
    }

    MPI_Finalize();
    return 0;
}
