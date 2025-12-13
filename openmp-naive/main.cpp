#include "openmp-naive.h"

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <N> [verify] [threads] [block_size]\n";
        std::cerr << "  N: Matrix size\n";
        std::cerr << "  verify: 0=skip, 1=verify (default: 0)\n";
        std::cerr << "  threads: Number of threads (default: max)\n";
        std::cerr << "  block_size: Block size (default: 128)\n";
        return 1;
    }

    int N = std::atoi(argv[1]);
    int check = (argc > 2) ? std::atoi(argv[2]) : 0;
    int num_threads = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();
    int block_size = (argc > 4) ? std::atoi(argv[4]) : 128;

    auto A = createRandomMatrix(N, 123);
    auto B = createRandomMatrix(N, 456);
    std::vector<float> C(N * N, 0.0f);

    Timer t;
    t.start();

    recursiveBlockedMatMul(N, A.data(), B.data(), C.data(), num_threads, block_size);

    float time = t.elapse();

    std::cout << "\nTotal execution time: " << time << " seconds\n";

    if (check)
    {
        std::cout << "\nVerifying Correctness...\n";

        Timer verify_timer;
        verify_timer.start();
        std::vector<float> CC(N * N, 0.0f);
        serialVerify(N, A.data(), B.data(), CC.data());
        float verify_time = verify_timer.elapse();

        std::cout << "Serial verification time: " << verify_time << "s\n";

        float diff_sum = 0.0, ref_sum = 0.0;
        int max_errors = 5, error_count = 0;

        for (int i = 0; i < N * N; ++i)
        {
            float diff = std::abs(C[i] - CC[i]);
            if (diff > 1e-3 && error_count < max_errors)
            {
                std::cout << "  Error at (" << i / N << "," << i % N << "): "
                          << "got " << C[i] << ", expected " << CC[i] << "\n";
                error_count++;
            }
            diff_sum += (C[i] - CC[i]) * (C[i] - CC[i]);
            ref_sum += CC[i] * CC[i];
        }

        float rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));

        if (rel_error < 1e-4)
            std::cout << "✓ PASSED\n";
        else
            std::cout << "✗ FAILED\n";

        std::cout << "Speedup vs Serial: " << std::fixed << std::setprecision(2)
                  << verify_time / time << "x\n";
    }

    return 0;
}