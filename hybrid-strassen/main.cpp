#include "hybrid-strassen.h"

int main(int argc, char **argv)
{
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (numProcs != 7)
    {
        if (rank == 0)
        {
            std::cerr << "Error: This implementation requires exactly 7 processes.\n";
        }
        MPI_Finalize();
        return 1;
    }

    if (argc < 3)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << " <matrix_size>" << "  <check err>  ";
        }
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);
    Timer timer;

    int k = THRESHOLD;
    int paddedSize = ((N + k - 1) / k) * k;

    int temp = paddedSize;
    while (temp > THRESHOLD)
    { // Limit depth
        temp /= 2;
    }

    if (rank == 0)
    {
        std::cout << "N=" << N << ", Padded=" << paddedSize
                  << ", Depth=" << MAX_DEPTH << std::endl;
    }

    std::vector<float> A, B, C(N * N, 0.0);
    if (paddedSize != N)
    {
        std::vector<float> A_padded, B_padded, C_padded;
        if (rank == 0)
        {

            A = createRandomMatrix(N, 123);
            B = createRandomMatrix(N, 456);

            A_padded.resize(paddedSize * paddedSize, 0.0);
            B_padded.resize(paddedSize * paddedSize, 0.0);
            C_padded.resize(paddedSize * paddedSize, 0.0);

#pragma omp parallel for collapse(2)
            for (int i = 0; i < N; ++i)
            {
                for (int j = 0; j < N; ++j)
                {
                    A_padded[i * paddedSize + j] = A[i * N + j];
                    B_padded[i * paddedSize + j] = B[i * N + j];
                }
            }
            A.clear();
            B.clear();
        }

        int m = paddedSize / 2;
        timer.start();
        std::vector<int> sendcounts(7, 0.0);
        sendcounts[0] = 0;
        sendcounts[1] = 4 * m * m;
        sendcounts[2] = 3 * m * m;
        sendcounts[3] = 3 * m * m;
        sendcounts[4] = 3 * m * m;
        sendcounts[5] = 3 * m * m;
        sendcounts[6] = 4 * m * m;

        std::vector<int> displs(7, 0);
        for (int i = 1; i < 7; i++)
        {
            displs[i] = sendcounts[i - 1] + displs[i - 1];
        }
        MPI_Bcast(sendcounts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(displs.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);

        int recvCount = sendcounts[rank];
        std::vector<float> recvbuf(recvCount, 0.0);

        strassen_mpi_wrapper(
            paddedSize,
            rank,
            numProcs,
            sendcounts.data(),
            displs.data(),
            A_padded.data(),
            paddedSize,
            B_padded.data(),
            paddedSize,
            C_padded.data(),
            paddedSize,
            recvCount,
            recvbuf.data(),
            &timer,
            MAX_DEPTH);

        if (rank == 0)
        {
#pragma omp parallel for
            for (int i = 0; i < N; ++i)
            {
                std::copy(C_padded.begin() + i * paddedSize, C_padded.begin() + i * paddedSize + N, C.begin() + i * N);
            }
        }
    }
    else
    {

        int m = N / 2;
        timer.start();
        std::vector<int> sendcounts(7, 0.0);
        sendcounts[0] = 0;
        sendcounts[1] = 4 * m * m;
        sendcounts[2] = 3 * m * m;
        sendcounts[3] = 3 * m * m;
        sendcounts[4] = 3 * m * m;
        sendcounts[5] = 3 * m * m;
        sendcounts[6] = 4 * m * m;

        std::vector<int> displs(7, 0);
        for (int i = 1; i < 7; i++)
        {
            displs[i] = sendcounts[i - 1] + displs[i - 1];
        }
        MPI_Bcast(sendcounts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(displs.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);

        int recvCount = sendcounts[rank];
        std::vector<float> recvbuf(recvCount, 0.0);

        if (rank == 0)
        {
            A = createRandomMatrix(N, 123);
            B = createRandomMatrix(N, 456);
        }

        strassen_mpi_wrapper(
            N,
            rank,
            numProcs,
            sendcounts.data(),
            displs.data(),
            A.data(),
            N,
            B.data(),
            N,
            C.data(),
            N,
            recvCount,
            recvbuf.data(),
            &timer,
            MAX_DEPTH);
    }

    if (rank == 0)
    {
        int check = std::atoi(argv[2]);
        if (check == 1)
        {

            std::vector<float> A, B;
            A = createRandomMatrix(N, 123);
            B = createRandomMatrix(N, 456);
            std::vector<float> CC(N * N, 0.0f);
            Timer naive_timer;
            naive_timer.start();
            naiveMultiply(N, A.data(), N, B.data(), N, CC.data(), N);
            float naiveTime = naive_timer.elapse();
            std::cout << "Naive completed in " << naiveTime << " seconds.\n";

            float diff_sum = 0.0, ref_sum = 0.0;
            for (int i = 0; i < N * N; ++i)
            {
                float d = C[i] - CC[i];
                diff_sum += d * d;
                ref_sum += CC[i] * CC[i];
            }
            float rel_error = std::sqrt(diff_sum / (ref_sum + 1e-12));
            std::cout << "Relative L2 error: " << rel_error << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}