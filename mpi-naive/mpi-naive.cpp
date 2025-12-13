#include "mpi-naive.h"

void initializeMatrices(int N, int rank, std::vector<int> &A, std::vector<int> &B, std::vector<int> &C)
{
    if (rank == 0)
    {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A[i * N + j] = 1 + std::rand() % 9;
                B[i * N + j] = 1 + std::rand() % 9;
            }
        }
    }
    else
    {
        B.resize(N * N);
    }
}

void distributeMatrices(int N, int rank, const std::vector<int> &A, std::vector<int> &local_a, std::vector<int> &B, int rows_per_proc)
{
    MPI_Scatter(
        (rank == 0 ? A.data() : nullptr),
        rows_per_proc * N,
        MPI_INT,
        local_a.data(),
        rows_per_proc * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    MPI_Bcast(
        B.data(),
        N * N,
        MPI_INT,
        0,
        MPI_COMM_WORLD);
}

void localMatrixComputation(int N, int rows_per_proc, const std::vector<int> &local_a, const std::vector<int> &B, std::vector<int> &local_c, double &local_time)
{
    double local_start = MPI_Wtime();

    for (int i = 0; i < rows_per_proc; i++)
    {
        for (int k = 0; k < N; k++)
        {
            int a_ik = local_a[i * N + k];
            for (int j = 0; j < N; j++)
            {
                local_c[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }

    double local_end = MPI_Wtime();
    local_time = local_end - local_start;
}

void cannonMatrixMultiply(int N, int rank, int size, const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, double& comp_time)
{
    int grid_size = (int)sqrt(size);
    if (grid_size * grid_size != size) {
        if (rank == 0) std::cout << "Cannon requires perfect square processes" << std::endl;
        return;
    }

    int block_size = N / grid_size;
    int row = rank / grid_size;
    int col = rank % grid_size;

    std::vector<int> local_A(block_size * block_size);
    std::vector<int> local_B(block_size * block_size);
    std::vector<int> local_C(block_size * block_size, 0);

    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            int global_i = row * block_size + i;
            int global_j = col * block_size + j;
            if (rank == 0) {
                local_A[i * block_size + j] = A[global_i * N + global_j];
                local_B[i * block_size + j] = B[global_i * N + global_j];
            }
        }
    }

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    int shift_source_A = (col - row + grid_size) % grid_size;
    int shift_dest_A = (col + row) % grid_size;
    MPI_Sendrecv_replace(local_A.data(), block_size * block_size, MPI_INT,
                         shift_dest_A, 0, shift_source_A, 0, row_comm, MPI_STATUS_IGNORE);

    int shift_source_B = (row - col + grid_size) % grid_size;
    int shift_dest_B = (row + col) % grid_size;
    MPI_Sendrecv_replace(local_B.data(), block_size * block_size, MPI_INT,
                         shift_dest_B, 0, shift_source_B, 0, col_comm, MPI_STATUS_IGNORE);

    double start = MPI_Wtime();
    
    for (int step = 0; step < grid_size; step++) {
        for (int i = 0; i < block_size; i++) {
            for (int k = 0; k < block_size; k++) {
                int a_ik = local_A[i * block_size + k];
                for (int j = 0; j < block_size; j++) {
                    local_C[i * block_size + j] += a_ik * local_B[k * block_size + j];
                }
            }
        }

        int left = (col - 1 + grid_size) % grid_size;
        int right = (col + 1) % grid_size;
        MPI_Sendrecv_replace(local_A.data(), block_size * block_size, MPI_INT,
                             left, 0, right, 0, row_comm, MPI_STATUS_IGNORE);

        int up = (row - 1 + grid_size) % grid_size;
        int down = (row + 1) % grid_size;
        MPI_Sendrecv_replace(local_B.data(), block_size * block_size, MPI_INT,
                             up, 0, down, 0, col_comm, MPI_STATUS_IGNORE);
    }
    
    comp_time = MPI_Wtime() - start;

    MPI_Gather(local_C.data(), block_size * block_size, MPI_INT,
               (rank == 0 ? C.data() : nullptr), block_size * block_size, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
}

void gatherResults(int N, int rank, int rows_per_proc, const std::vector<int> &local_c, std::vector<int> &C)
{
    MPI_Gather(local_c.data(), rows_per_proc * N, MPI_INT, (rank == 0 ? C.data() : nullptr), rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);
}

double computeMaxLocalTime(double local_time, int /* rank */)
{
    double max_local_time = 0.0;
    MPI_Reduce(&local_time, &max_local_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_local_time;
}

void serialVerify(int N, const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C_verify)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += A[i * N + k] * B[k * N + j];
            }
            C_verify[i * N + j] = sum;
        }
    }
}

bool verifyResults(int N, const std::vector<int>& C, const std::vector<int>& C_verify, int rank)
{
    if (rank != 0)
        return true; // Only rank 0 verifies

    long long diff_sum = 0;
    long long ref_sum = 0;
    int max_errors_to_show = 5;
    int error_count = 0;

    for (int i = 0; i < N * N; ++i)
    {
        long long diff = std::abs(static_cast<long long>(C[i]) - static_cast<long long>(C_verify[i]));
        if (diff > 0 && error_count < max_errors_to_show)
        {
            int row = i / N;
            int col = i % N;
            std::cout << "  Error at (" << row << "," << col << "): "
                      << "got " << C[i] << ", expected " << C_verify[i]
                      << ", diff=" << diff << std::endl;
            error_count++;
        }
        diff_sum += diff * diff;
        ref_sum += static_cast<long long>(C_verify[i]) * static_cast<long long>(C_verify[i]);
    }

    double rel_error = std::sqrt(static_cast<double>(diff_sum) / (static_cast<double>(ref_sum) + 1e-12));
    std::cout << "\nRelative L2 error: " << std::scientific << rel_error << std::endl;

    if (error_count > 0)
    {
        std::cout << "Total errors found: " << error_count;
        if (error_count >= max_errors_to_show)
            std::cout << " (showing first " << max_errors_to_show << ")";
        std::cout << std::endl;
    }

    bool passed = (rel_error < 1e-6);
    if (passed)
    {
        std::cout << "✓ PASSED - Results are correct!" << std::endl;
    }
    else
    {
        std::cout << "✗ FAILED - Results differ significantly!" << std::endl;
    }

    return passed;
}
