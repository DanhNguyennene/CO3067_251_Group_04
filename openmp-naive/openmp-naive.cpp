#include "openmp-naive.h"

std::vector<float> createRandomMatrix(int size, int seed)
{
    std::vector<float> matrix(size * size);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(LOWER_B, UPPER_B);
    for (int i = 0; i < size * size; ++i)
    {
        matrix[i] = dist(rng);
    }
    return matrix;
}

void serialVerify(int n, const float *A, const float *B, float *C)
{
    std::fill(C, C + n * n, 0.0f);
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            float a_ik = A[i * n + k];
            for (int j = 0; j < n; ++j)
            {
                C[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }
}

void naiveAddMultiply(
    int n,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc)
{
    for (int i = 0; i < n; ++i)
    {
        for (int k = 0; k < n; ++k)
        {
            float a_ik = A[i * lda + k];

#pragma omp simd
            for (int j = 0; j < n; ++j)
            {
                C[i * ldc + j] += a_ik * B[k * ldb + j];
            }
        }
    }
}

void tiledMatMul(
    int n,
    const float *A,
    const float *B,
    float *C,
    int num_threads,
    int tile_size)
{
    omp_set_num_threads(num_threads);
    std::fill(C, C + n * n, 0.0f);

#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ii = 0; ii < n; ii += tile_size)
    {
        for (int jj = 0; jj < n; jj += tile_size)
        {
            for (int kk = 0; kk < n; kk += tile_size)
            {
                int i_end = std::min(ii + tile_size, n);
                int j_end = std::min(jj + tile_size, n);
                int k_end = std::min(kk + tile_size, n);

                for (int i = ii; i < i_end; ++i)
                {
                    for (int k = kk; k < k_end; ++k)
                    {
                        float a_ik = A[i * n + k];
#pragma omp simd
                        for (int j = jj; j < j_end; ++j)
                        {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

void recursiveMatMul(
    int n,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int threshold)
{
    // Handle non-power-of-2 sizes and small matrices
    if (n <= threshold)
    {
        naiveAddMultiply(n, A, lda, B, ldb, C, ldc);
        return;
    }

    // For odd sizes, use naive multiplication
    if (n % 2 != 0)
    {
        naiveAddMultiply(n, A, lda, B, ldb, C, ldc);
        return;
    }

    int m = n / 2;

    const float *A11 = A;
    const float *A12 = A + m;
    const float *A21 = A + m * lda;
    const float *A22 = A + m * lda + m;

    const float *B11 = B;
    const float *B12 = B + m;
    const float *B21 = B + m * ldb;
    const float *B22 = B + m * ldb + m;

    float *C11 = C;
    float *C12 = C + m;
    float *C21 = C + m * ldc;
    float *C22 = C + m * ldc + m;

#pragma omp taskgroup
    {
// First wave: C11 += A11*B11, C12 += A11*B12, C21 += A21*B11, C22 += A21*B12
#pragma omp task
        recursiveMatMul(m, A11, lda, B11, ldb, C11, ldc, threshold);

#pragma omp task
        recursiveMatMul(m, A11, lda, B12, ldb, C12, ldc, threshold);

#pragma omp task
        recursiveMatMul(m, A21, lda, B11, ldb, C21, ldc, threshold);

#pragma omp task
        recursiveMatMul(m, A21, lda, B12, ldb, C22, ldc, threshold);

#pragma omp taskwait

// Second wave: C11 += A12*B21, C12 += A12*B22, C21 += A22*B21, C22 += A22*B22
#pragma omp task
        recursiveMatMul(m, A12, lda, B21, ldb, C11, ldc, threshold);

#pragma omp task
        recursiveMatMul(m, A12, lda, B22, ldb, C12, ldc, threshold);

#pragma omp task
        recursiveMatMul(m, A22, lda, B21, ldb, C21, ldc, threshold);

#pragma omp task
        recursiveMatMul(m, A22, lda, B22, ldb, C22, ldc, threshold);
    }
}

void parallelDCMatMul(
    int n,
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int num_threads,
    int threshold)
{
    omp_set_num_threads(num_threads);

#pragma omp parallel
    {
#pragma omp single
        {
            recursiveMatMul(n, A.data(), n, B.data(), n, C.data(), n, threshold);
        }
    }
}

// Pad matrix to nearest power of 2 (optional enhancement)
int nextPowerOf2(int n)
{
    int power = 1;
    while (power < n)
        power *= 2;
    return power;
}

void printBenchmarkHeader()
{
    std::cout << "\n================================================" << std::endl;
    std::cout << "OpenMP Divide & Conquer Matrix Multiplication" << std::endl;
    std::cout << "================================================" << std::endl;
}

void printResults(int n, int threads, int threshold, float time, bool show_gflops)
{
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Size: " << n << "x" << n
              << " | Threads: " << threads
              << " | Threshold: " << threshold
              << " | Time: " << time << "s";

    if (show_gflops)
    {
        double flops = 2.0 * n * n * n;
        double gflops = (flops / time) / 1e9;
        std::cout << " | " << std::setprecision(2) << gflops << " GFLOPS";
    }
    std::cout << std::endl;
}