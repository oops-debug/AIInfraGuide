
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call) do {                                \
    cudaError_t e = (call);                             \
    if (e != cudaSuccess) {                             \
        fprintf(stderr, "%s\n",                         \
                cudaGetErrorString(e));                 \
        exit(1);                                        \
    }                                                   \
} while (0)

constexpr int TILE = 32;
constexpr int ROWS = 8;
constexpr int N = 2048;

// PAD=0: tile[32][32]
// PAD=1: tile[32][33]
template<int PAD>
__global__ void transpose(
    const float* input,
    float* output,
    int width)
{
    __shared__ float tile[TILE][TILE + PAD];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 第一阶段：从全局内存读取
    int x = blockIdx.x * TILE + tx;
    int y = blockIdx.y * TILE + ty;

    for (int j = 0; j < TILE; j += ROWS) {
        tile[ty + j][tx] =
            input[(y + j) * width + x];
    }

    __syncthreads();

    // 第二阶段：转置后写回全局内存
    x = blockIdx.y * TILE + tx;
    y = blockIdx.x * TILE + ty;

    for (int j = 0; j < TILE; j += ROWS) {
        output[(y + j) * width + x] =
            tile[tx][ty + j];
    }
}

template<int PAD>
float benchmark(
    const float* input,
    float* output)
{
    dim3 block(TILE, ROWS);
    dim3 grid(N / TILE, N / TILE);

    // Warmup
    for (int i = 0; i < 10; ++i) {
        transpose<PAD><<<grid, block>>>(
            input, output, N);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));

    for (int i = 0; i < 100; ++i) {
        transpose<PAD><<<grid, block>>>(
            input, output, N);
    }

    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return ms / 100.0f;
}

bool verify(const std::vector<float>& out,
            const std::vector<float>& in)
{
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            if (out[x * N + y] != in[y * N + x])
                return false;
        }
    }
    return true;
}

int main()
{
    size_t count = size_t(N) * N;
    size_t bytes = count * sizeof(float);

    std::vector<float> input(count);
    std::vector<float> output(count);

    for (size_t i = 0; i < count; ++i)
        input[i] = float(i % 997);

    float *d_input, *d_output;
    CHECK(cudaMalloc(&d_input, bytes));
    CHECK(cudaMalloc(&d_output, bytes));

    CHECK(cudaMemcpy(d_input, input.data(),
                     bytes, cudaMemcpyHostToDevice));

    float t32 = benchmark<0>(d_input, d_output);

    CHECK(cudaMemcpy(output.data(), d_output,
                     bytes, cudaMemcpyDeviceToHost));
    bool ok32 = verify(output, input);

    float t33 = benchmark<1>(d_input, d_output);

    CHECK(cudaMemcpy(output.data(), d_output,
                     bytes, cudaMemcpyDeviceToHost));
    bool ok33 = verify(output, input);

    printf("tile[32][32]: %.4f ms, %s\n",
           t32, ok32 ? "PASS" : "FAIL");
    printf("tile[32][33]: %.4f ms, %s\n",
           t33, ok33 ? "PASS" : "FAIL");

    printf("Speedup: %.2fx\n", t32 / t33);

    CHECK(cudaFree(d_input));
    CHECK(cudaFree(d_output));
}
