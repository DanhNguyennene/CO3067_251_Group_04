#ifndef HYBRID_STRASSEN_H
#define HYBRID_STRASSEN_H

#include <mpi.h>
#include <iostream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>

#define LOWER_B 0.0
#define UPPER_B 1.0

#define THRESHOLD 128
#define MAX_DEPTH 4

class Timer
{
    std::chrono::high_resolution_clock::time_point start_;

public:
    void start()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }
    float elapse()
    {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float>(end - start_).count();
    }
};

// Matrix utilities
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

// Strassen multiplication
void strassenSerial(
    int n, const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    float *work);

void strassenParallel(
    int n,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int depth,
    int max_depth);

// MPI wrapper for Strassen
void strassen_mpi_wrapper(
    int N,
    int rank,
    int numProcs,
    int *sendcounts,
    int *displs,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int recvCount,
    float *recvbuf,
    Timer *timer,
    int max_depth);

#endif
