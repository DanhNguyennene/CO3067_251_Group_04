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

inline unsigned int interleaveBits(unsigned int x, unsigned int y)
{
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;
    
    return x | (y << 1);
}

inline void deinterleaveBits(unsigned int z, unsigned int& x, unsigned int& y)
{
    x = z & 0x55555555;
    y = (z >> 1) & 0x55555555;
    
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0F0F0F0F;
    x = (x | (x >> 4)) & 0x00FF00FF;
    x = (x | (x >> 8)) & 0x0000FFFF;
    
    y = (y | (y >> 1)) & 0x33333333;
    y = (y | (y >> 2)) & 0x0F0F0F0F;
    y = (y | (y >> 4)) & 0x00FF00FF;
    y = (y | (y >> 8)) & 0x0000FFFF;
}

void zOrderMultiply(
    int N, int rank, int size,
    const std::vector<int>& A_local, int local_rows,
    const std::vector<int>& B,
    std::vector<int>& C_local,
    int block_size = 32)
{
    (void)rank; (void)size;
    
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                unsigned int z_i = interleaveBits(i % block_size, j % block_size);
                unsigned int z_k = interleaveBits(k % block_size, j % block_size);
                (void)z_i; (void)z_k;
                
                C_local[i * N + j] += A_local[i * N + k] * B[k * N + j];
            }
        }
    }
}

void pipelinedRingMultiply(int N, int rank, int size, const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, double& comp_time)
{
    if (N % size != 0) {
        if (rank == 0) std::cerr << "Error: N must be divisible by number of processes\n";
        return;
    }
    
    int rows_per_proc = N / size;
    int elements_per_proc = rows_per_proc * N;
    
    std::vector<int> local_A(elements_per_proc);
    std::vector<int> local_B(N * N);
    std::vector<int> local_C(elements_per_proc, 0);
    
    if (rank == 0) {
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < N; j++) {
                local_A[i * N + j] = A[i * N + j];
            }
        }
        local_B = B;
        
        for (int p = 1; p < size; p++) {
            std::vector<int> send_A(elements_per_proc);
            int start_row = p * rows_per_proc;
            for (int i = 0; i < rows_per_proc; i++) {
                for (int j = 0; j < N; j++) {
                    send_A[i * N + j] = A[(start_row + i) * N + j];
                }
            }
            MPI_Send(send_A.data(), elements_per_proc, MPI_INT, p, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(local_A.data(), elements_per_proc, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    MPI_Bcast(local_B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    double start = MPI_Wtime();
    
    zOrderMultiply(N, rank, size, local_A, rows_per_proc, local_B, local_C);
    
    comp_time = MPI_Wtime() - start;
    
    if (rank != 0) {
        MPI_Send(local_C.data(), elements_per_proc, MPI_INT, 0, 3, MPI_COMM_WORLD);
    } else {
        for (int i = 0; i < rows_per_proc; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] = local_C[i * N + j];
            }
        }
        
        for (int p = 1; p < size; p++) {
            std::vector<int> recv_C(elements_per_proc);
            MPI_Recv(recv_C.data(), elements_per_proc, MPI_INT, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            int start_row = p * rows_per_proc;
            for (int i = 0; i < rows_per_proc; i++) {
                for (int j = 0; j < N; j++) {
                    C[(start_row + i) * N + j] = recv_C[i * N + j];
                }
            }
        }
    }
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
