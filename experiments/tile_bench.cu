#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

#define N 2048      // A+B+C 共 48MB，超出 L2（几十MB），DRAM 流量真实

// ❌ Naive：C = A×B，B 按列访问 → warp 内大 stride
__global__ void matmulNaive(const float* __restrict__ A, const float* __restrict__ B,
                            float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float s = 0.0f;
        for (int k = 0; k < n; k++)
            s += A[row * n + k] * B[k * n + col];   // A 合并，B 按列：相邻线程差 n*4B
        C[row * n + col] = s;
    }
}

// ✅ Tiled：分块加载到共享内存，块内复用 + 合并访问（TILE 做成模板参数，方便扫描 8/16/32）
template <int TILE>
__global__ void matmulTiled(const float* __restrict__ A, const float* __restrict__ B,
                            float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float s = 0.0f;

    for (int t = 0; t < n / TILE; t++) {
        // 协作加载：warp 内相邻线程读连续地址（合并访问）
        As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++)
            s += As[threadIdx.y][k] * Bs[k][threadIdx.x];  // 每个数据被 Block 内 32 线程复用
        __syncthreads();
    }
    C[row * n + col] = s;
}

template <typename F>
float bench(F&& launch) {
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    launch(); cudaDeviceSynchronize();              // 预热

    float best = 1e9f;
    for (int k = 0; k < 5; k++) {                   // 取 5 次最优，避开时钟抖动
        cudaEventRecord(t0);
        launch();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return best;
}

int main() {
    size_t bytes = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes), *h_B = (float*)malloc(bytes);
    float *d_A, *d_B, *d_C1, *d_C2;
    cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C1, bytes); cudaMalloc(&d_C2, bytes);

    for (size_t i = 0; i < (size_t)N * N; i++) {    // 取值小，控制浮点累加误差
        h_A[i] = (i % 7) * 0.01f;
        h_B[i] = ((i / 13) % 5) * 0.01f;
    }
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 blockN(16, 16), gridN(N / 16, N / 16);

    float msNaive = bench([&]{ matmulNaive<<<gridN, blockN>>>(d_A, d_B, d_C1, N); });
    // TILE 扫描：8 / 16 / 32（最后一次写 C2，用于正确性比对）
    float ms8  = bench([&]{ matmulTiled<8> <<<dim3(N / 8,  N / 8),  dim3(8,  8) >>>(d_A, d_B, d_C2, N); });
    float ms16 = bench([&]{ matmulTiled<16><<<dim3(N / 16, N / 16), dim3(16, 16)>>>(d_A, d_B, d_C2, N); });
    float ms32 = bench([&]{ matmulTiled<32><<<dim3(N / 32, N / 32), dim3(32, 32)>>>(d_A, d_B, d_C2, N); });

    double gflop = 2.0 * N * N * N / 1e9;
    printf("N=%d (%.1f GFLOP/次)\n\n", N, gflop);
    printf("%-10s %12s %16s %14s\n", "版本", "耗时(ms)", "性能", "vs Naive");
    printf("%-10s %12.2f %13.1f GFLOPS %9.2fx\n", "Naive",   msNaive, gflop / msNaive * 1e3, 1.0);
    printf("%-10s %12.2f %13.1f GFLOPS %9.2fx\n", "Tile=8",  ms8,     gflop / ms8     * 1e3, msNaive / ms8);
    printf("%-10s %12.2f %13.1f GFLOPS %9.2fx\n", "Tile=16", ms16,    gflop / ms16    * 1e3, msNaive / ms16);
    printf("%-10s %12.2f %13.1f GFLOPS %9.2fx\n", "Tile=32", ms32,    gflop / ms32    * 1e3, msNaive / ms32);

    // 正确性校验：拷回 CPU 逐元素比对（两个 kernel 累加顺序相同，理论上应逐位一致）
    float *h_C1 = (float*)malloc(bytes), *h_C2 = (float*)malloc(bytes);
    cudaMemcpy(h_C1, d_C1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, bytes, cudaMemcpyDeviceToHost);
    double maxDiff = 0.0;
    for (size_t i = 0; i < (size_t)N * N; i++) {
        double d = fabs((double)h_C1[i] - h_C2[i]);
        if (d > maxDiff) maxDiff = d;
    }
    printf("正确性: max|C1-C2| = %.6f %s\n", maxDiff, maxDiff < 0.01 ? "(PASS)" : "(FAIL!)");

    free(h_A); free(h_B); free(h_C1); free(h_C2);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1); cudaFree(d_C2);
    return 0;
}