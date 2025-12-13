#ifndef OPENMP_NAIVE_H
#define OPENMP_NAIVE_H

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <omp.h>
#include <cstring>
#include <algorithm>
#include <iomanip>

#define LOWER_B 0.0
#define UPPER_B 1.0

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

std::vector<float> createRandomMatrix(int size, int seed);

void serialVerify(int n, const float *A, const float *B, float *C);

void naiveAddMultiply(
    int n,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc);

void tiledMatMul(
    int n,
    const float *A,
    const float *B,
    float *C,
    int num_threads,
    int tile_size);

void recursiveBlockedMatMul(
    int n,
    const float *A,
    const float *B,
    float *C,
    int num_threads,
    int block_size);

void recursiveMatMul(
    int n,
    const float *A, int lda,
    const float *B, int ldb,
    float *C, int ldc,
    int threshold);

void parallelDCMatMul(
    int n,
    const std::vector<float> &A,
    const std::vector<float> &B,
    std::vector<float> &C,
    int num_threads,
    int threshold);

int nextPowerOf2(int n);

void printBenchmarkHeader();

void printResults(int n, int threads, int threshold, float time, bool show_gflops = false);

#endif
