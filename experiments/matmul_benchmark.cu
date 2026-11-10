
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef TILE
#define TILE 16
#endif

#define CHECK(call) do {                                      \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",          \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)

// ======================================
// 1. Naive MatMul
// 每个线程计算 C 中的一个元素
// ======================================
__global__ void matmulNaive(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// ======================================
// 2. Tiled MatMul
// 一个 Block 协作加载 Shared Memory
// ======================================
__global__ void matmulTiled(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {

        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // 加载 A 的子矩阵
        if (row < M && aCol < K)
            As[ty][tx] = A[row * K + aCol];
        else
            As[ty][tx] = 0.0f;

        // 加载 B 的子矩阵
        if (bRow < K && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        // 确保所有线程完成加载
        __syncthreads();

        // Shared Memory 中完成 TILE 次乘加
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // 防止下一轮加载覆盖正在读取的数据
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ======================================
// 3. CUDA Event 性能测量
// ======================================
float benchmark(
    bool tiled,
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE,
              (M + TILE - 1) / TILE);

    auto launch = [&]() {
        if (tiled)
            matmulTiled<<<grid, block>>>(A, B, C, M, N, K);
        else
            matmulNaive<<<grid, block>>>(A, B, C, M, N, K);
    };

    // Warmup
    for (int i = 0; i < 5; i++)
        launch();

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    constexpr int REPEAT = 20;

    CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEAT; i++)
        launch();

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaGetLastError());

    float elapsed = 0.0f;
    CHECK(cudaEventElapsedTime(&elapsed, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return elapsed / REPEAT;
}

// ======================================
// 4. 单组矩阵测试
// ======================================
void runTest(int M, int N, int K)
{
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K);
    std::vector<float> hB((size_t)K * N);

    // 固定输入，结果可复现
    for (size_t i = 0; i < hA.size(); i++)
        hA[i] = (int)(i % 17 - 8) * 0.01f;

    for (size_t i = 0; i < hB.size(); i++)
        hB[i] = (int)(i % 13 - 6) * 0.01f;

    // 注意：size_t 无符号减法必须先转换
    // 上述表达式中的转换需放在减法之前

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytesA));
    CHECK(cudaMalloc(&dB, bytesB));
    CHECK(cudaMalloc(&dC, bytesC));

    CHECK(cudaMemcpy(dA, hA.data(), bytesA,
                     cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB.data(), bytesB,
                     cudaMemcpyHostToDevice));

    // Naive
    float naiveMs = benchmark(false, dA, dB, dC, M, N, K);

    std::vector<float> naiveOut((size_t)M * N);
    CHECK(cudaMemcpy(naiveOut.data(), dC, bytesC,
                     cudaMemcpyDeviceToHost));

    // Tiled
    float tiledMs = benchmark(true, dA, dB, dC, M, N, K);

    std::vector<float> tiledOut((size_t)M * N);
    CHECK(cudaMemcpy(tiledOut.data(), dC, bytesC,
                     cudaMemcpyDeviceToHost));

    // 数值正确性比较
    double maxDiff = 0.0;
    for (size_t i = 0; i < naiveOut.size(); i++) {
        maxDiff = std::max(
            maxDiff,
            (double)std::fabs(naiveOut[i] - tiledOut[i])
        );
    }

    double flops = 2.0 * M * N * K;
    double naiveGFLOPS = flops / (naiveMs * 1e6);
    double tiledGFLOPS = flops / (tiledMs * 1e6);

    printf("%5d %5d %5d | %9.4f %9.4f | "
           "%7.2fx | %9.1f %9.1f | %.2e\n",
           M, N, K,
           naiveMs, tiledMs,
           naiveMs / tiledMs,
           naiveGFLOPS, tiledGFLOPS,
           maxDiff);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dB));
    CHECK(cudaFree(dC));
}

int main()
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("TILE = %d\n\n", TILE);

    printf("    M     N     K | "
           "Naive(ms) Tiled(ms) | Speedup | "
           "Naive GF  Tiled GF | MaxDiff\n");

    runTest(256, 256, 256);
    runTest(512, 512, 512);
    runTest(1024, 1024, 1024);
    runTest(2048, 2048, 2048);

    // 非 TILE 整倍数，检查边界逻辑
    runTest(1003, 1007, 509);

    return 0;
}

