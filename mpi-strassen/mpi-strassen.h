#ifndef MPI_STRASSEN_H
#define MPI_STRASSEN_H

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <cstring>

// Random number generation bounds
#define LOWER_B 0.0
#define UPPER_B 1.0
#define THRESHOLD 128

// =======================
// Timer class
// =======================
class Timer
{
    std::chrono::high_resolution_clock::time_point start_;

public:
    void start() { start_ = std::chrono::high_resolution_clock::now(); }
    float elapse()
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float>(end - start_).count();
    }
};

// =======================
// Matrix utilities
// =======================
std::vector<float> createRandomMatrix(int size, int seed);

void naiveMultiply(
    int n,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc);

void addMatrix(
    int n, const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc);

void subtractMatrix(
    int n, const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc);

void strassenSerial(
    int n,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    float *work);

void strassen_mpi_wrapper(
    int N,
    int rank,
    int numProcs,
    int *sendcounts,
    int *displs,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc,
    int recvCount,
    float *recvbuf,
    Timer *timer);

#endif
