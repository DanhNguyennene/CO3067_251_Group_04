#ifndef OPENMP_STRASSEN_H
#define OPENMP_STRASSEN_H

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <iomanip>

// Random number generation bounds
#define LOWER_B 0.0
#define UPPER_B 1.0

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

void serialVerify(int n, const float *A, const float *B, float *C);

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
    float *work,
    int threshold);

void strassenParallel(
    int n,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int depth,
    int max_depth,
    int threshold);

void strassenMatMul(
    int n,
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int num_threads = 0,
    int threshold = 128,
    int max_depth = 4);

int nextPowerOf2(int n);

void printBenchmarkHeader();

void printResults(
    int n,
    int threads,
    int threshold,
    int max_depth,
    int padded,
    float time,
    bool show_gflops = true);

#endif
