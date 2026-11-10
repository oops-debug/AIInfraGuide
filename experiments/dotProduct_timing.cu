// ============================================================
// 练习 8.1 延伸：向量点积 CPU vs GPU 耗时 + 精度对比（N 扫描版）
//
// 目的：
//   1) 看 GPU 相对 CPU 的加速比如何随数据量变化
//   2) 看清一个反直觉事实：N 大到一定程度，GPU 反而比 CPU 更准
//
// ⚠️ 重要结论（实测得出）：
//   CPU 用 float 严格串行累加时，sum 涨到 2^24 = 16777216 就彻底饱和——
//   该量级 ulp 变成 2，加 1 落在两个可表示值中点，ties-to-even 舍回原值，
//   之后每加一次都原地打转。所以 N=32M/64M 时 CPU 结果停在 16777216。
//   GPU 的树形规约每层都加同量级的数（块内全是 1、跨块每次加 256，
//   而 256 在 3.3e7/6.7e7 量级上都是 ulp 的整数倍），结果精确等于 N。
//
//   因此：误差判定必须对着 double 真值，不能拿 CPU 当基准。
//
// 计时方法：
//   CPU：std::chrono 单次（可自行改成取 3 次最优，加速比会更稳）
//   GPU：预热 1 次 + 跑 10 次取最优（每轮必须 memset 清零 result）
//
// 编译：nvcc -O3 -arch=sm_120 dotProduct_timing.cu -o dotProduct_timing
//       Windows 原生需追加 -Xcompiler "/utf-8"（源码含中文注释）
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define THREADS 256
#define REPS 10        // GPU 计时重复次数，取最优

// 未完成点积练习前请先别看这段（自己补完再看更有效）
__global__ void dotProduct(const float* __restrict__ A, const float* __restrict__ B,
                           float* result, int n) {
    __shared__ float sdata[THREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? A[idx] * B[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(result, sdata[0]);
}

// 跑一个规模：CPU/GPU 计时 + 精度（都以 double 为真值）
static void runSize(int n) {
    size_t bytes = (size_t)n * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // ---------- 真值：double 累加（不受 float ulp 限制） ----------
    double ref = 0.0;
    for (int i = 0; i < n; i++) {
        ref += (double)h_A[i] * (double)h_B[i];
    }

    // ---------- CPU float 串行点积（计时对象） ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    float cpuResult = 0.0f;
    for (int i = 0; i < n; i++) {
        cpuResult += h_A[i] * h_B[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- GPU 点积 ----------
    float *d_A, *d_B, *d_result;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int grid = (n + THREADS - 1) / THREADS;

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    // 预热：首次初始化 + 拉高时钟
    cudaMemset(d_result, 0, sizeof(float));
    dotProduct<<<grid, THREADS>>>(d_A, d_B, d_result, n);
    cudaDeviceSynchronize();

    float bestMs = 1e9f;
    for (int k = 0; k < REPS; k++) {
        cudaMemset(d_result, 0, sizeof(float));   // 每轮清零，否则结果累加 REPS 次
        cudaEventRecord(e0);
        dotProduct<<<grid, THREADS>>>(d_A, d_B, d_result, n);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < bestMs) bestMs = ms;
    }

    float gpuResult;
    cudaMemcpy(&gpuResult, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // 事件只包住 kernel，不含 memcpy —— 纯计算口径
    double gbps   = (double)bytes * 2 / (bestMs * 1e6);
    double errCpu = fabs((double)cpuResult - ref) / ref;
    double errGpu = fabs((double)gpuResult - ref) / ref;

    printf("%9d %7.1f MB %9.3f %9.3f %8.1fx %8.1f %10.2e %10.2e%s\n",
           n, (double)bytes * 2 / (1024 * 1024),
           cpuMs, bestMs, cpuMs / bestMs, gbps, errCpu, errGpu,
           (errGpu < 1e-6 && errCpu > 1e-3) ? "  <-- CPU 饱和" : "");

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_result);
    free(h_A); free(h_B);
}

int main() {
    // 输入全填 1.0：double 真值 = N，便于交叉验证
    int sizes[] = {1 << 20, 1 << 21, 1 << 22, 1 << 23, 1 << 24, 1 << 25, 1 << 26};
    const int count = sizeof(sizes) / sizeof(sizes[0]);

    printf("GPU: RTX 5060 Laptop (theoretical BW 384 GB/s)\n");
    printf("note: GPU col = CUDA Event kernel time (best of %d, warmed up), memcpy excluded\n", REPS);
    printf("      RelErr computed against double-precision reference\n\n");
    printf("%9s %10s %9s %9s %9s %8s %11s %11s\n",
           "N", "Bytes", "CPU(ms)", "GPU(ms)", "Speedup", "GB/s", "CPU err", "GPU err");
    printf("%s\n",
           "--------------------------------------------------------------------------------------------");

    for (int i = 0; i < count; i++) {
        runSize(sizes[i]);
    }

    printf("\n提示: CPU err 突然跳到 1e-1 量级 = float 串行累加在 2^24 处饱和（见下方结论）;\n");
    printf("      GPU GB/s 贴近 384 说明合并访问接近吃满（对照 1.3 节 3.5 实验的 346 GB/s）\n");
    return 0;
}
