
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do {                                    \
    cudaError_t err = (call);                               \
    if (err != cudaSuccess) {                               \
        fprintf(stderr, "CUDA error: %s\n",                  \
                cudaGetErrorString(err));                   \
        exit(1);                                            \
    }                                                       \
} while (0)

__global__ void strided_read(
    const float* input,
    float* output,
    size_t N,
    int stride)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] = input[i * stride];
    }
}

int main()
{
    const size_t N = 1 << 22;
    const int blockSize = 256;
    const int repeats = 30;
    const int strides[] = {1, 2, 4, 8, 16, 32};

    const size_t maxStride = 32;
    const size_t inputBytes =
        ((N - 1) * maxStride + 1) * sizeof(float);
    const size_t outputBytes = N * sizeof(float);

    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK(cudaMalloc(&d_input, inputBytes));
    CHECK(cudaMalloc(&d_output, outputBytes));

    CHECK(cudaMemset(d_input, 0, inputBytes));

    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    printf("Stride, Time(ms), Effective GB/s\n");

    for (int stride : strides) {

        // 预热
        for (int r = 0; r < 5; ++r) {
            strided_read<<<gridSize, blockSize>>>(
                d_input, d_output, N, stride);
        }
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());

        CHECK(cudaEventRecord(start));

        for (int r = 0; r < repeats; ++r) {
            strided_read<<<gridSize, blockSize>>>(
                d_input, d_output, N, stride);
        }

        CHECK(cudaGetLastError());
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= repeats;

        // 有效读取 + 有效写入
        double bytes = 2.0 * N * sizeof(float);

        // GB/s，按十进制计算
        double bandwidth = bytes / (ms * 1e6);

        printf("%2d, %.4f, %.2f\n",
               stride, ms, bandwidth);
    }

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));

    return 0;
}
