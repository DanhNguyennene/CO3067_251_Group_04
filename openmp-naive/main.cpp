#include "openmp-naive.h"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> [check_error] [num_threads] [threshold]" << std::endl;
        std::cerr << "  matrix_size: Size of square matrix (e.g., 1000)" << std::endl;
        std::cerr << "  check_error: 0=skip, 1=verify (default: 1)" << std::endl;
        std::cerr << "  num_threads: Number of OpenMP threads (default: max)" << std::endl;
        std::cerr << "  threshold: Base case size (default: 128)" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    int check = (argc > 2) ? std::atoi(argv[2]) : 1;
    int num_threads = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();
    int threshold = (argc > 4) ? std::atoi(argv[4]) : 128;

    printBenchmarkHeader();
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
    std::cout << "Max threads available: " << omp_get_max_threads() << std::endl;

    // Warn about non-power-of-2 sizes
    int next_pow2 = nextPowerOf2(N);
    if (next_pow2 != N)
    {
        std::cout << "⚠ Warning: Size " << N << " is not a power of 2." << std::endl;
        std::cout << "  Algorithm will fall back to naive multiplication for odd-sized blocks." << std::endl;
        std::cout << "  Nearest power of 2: " << next_pow2 << std::endl;
    }
    std::cout << "================================================" << std::endl;

    std::cout << "\nInitializing matrices..." << std::endl;
    auto A = createRandomMatrix(N, 123);
    auto B = createRandomMatrix(N, 456);
    std::vector<float> C(N * N, 0.0f);

    std::cout << "Starting Parallel Divide & Conquer..." << std::endl;
    Timer t;
    t.start();

    parallelDCMatMul(N, A, B, C, num_threads, threshold);

    float time = t.elapse();

    std::cout << "\n================================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "================================================" << std::endl;
    printResults(N, num_threads, threshold, time, false);

    if (check == 0)
    {
        std::cout << "================================================" << std::endl;
        return 0;
    }

    std::cout << "\n================================================" << std::endl;
    std::cout << "Verifying Correctness..." << std::endl;
    std::cout << "================================================" << std::endl;

    Timer verify_timer;
    verify_timer.start();
    std::vector<float> CC(N * N, 0.0f);
    serialVerify(N, A.data(), B.data(), CC.data());
    float verify_time = verify_timer.elapse();

    std::cout << "Serial verification time: " << verify_time << "s" << std::endl;

    float diff_sum = 0.0, ref_sum = 0.0;
    int max_errors_to_show = 5;
    int error_count = 0;

    for (int i = 0; i < N * N; ++i)
    {
        float diff = std::abs(C[i] - CC[i]);
        if (diff > 1e-3 && error_count < max_errors_to_show)
        {
            int row = i / N;
            int col = i % N;
            std::cout << "  Error at (" << row << "," << col << "): "
                      << "got " << C[i] << ", expected " << CC[i]
                      << ", diff=" << diff << std::endl;
            error_count++;
        }
        diff_sum += (C[i] - CC[i]) * (C[i] - CC[i]);
        ref_sum += CC[i] * CC[i];
    }

    float rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));
    std::cout << "\nRelative L2 error: " << std::scientific << rel_error << std::endl;

    if (rel_error < 1e-4)
    {
        std::cout << "✓ PASSED - Results are correct!" << std::endl;
    }
    else
    {
        std::cout << "✗ FAILED - Results differ significantly!" << std::endl;
    }

    std::cout << "\nSpeedup vs Serial: " << std::fixed << std::setprecision(2)
              << verify_time / time << "x" << std::endl;
    std::cout << "================================================" << std::endl;

    return 0;
}