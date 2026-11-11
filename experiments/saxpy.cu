// ============================================================
// 练习 8.2：SAXPY —— y = a * x + y
//
// 和 1.4 的 vectorAdd（c = a + b）相比，SAXPY 有三个新特点：
//   1) 标量参数 a 要传进 kernel（kernel 参数可以是普通标量，传值）
//   2) 是"读-改-写"同一个数组（in-place），不存在第三个输出数组
//   3) 每个元素访存 3 次（读 x、读 y、写 y）= 12 字节，而 vectorAdd 是 12 字节也是 3 次
//      → 但 SAXPY 的 y 既读又写，缓存行为和 vectorAdd 不完全一样，值得测带宽
//
// 你的任务：补全 4 处 TODO（kernel 2 处 + main 2 处）+ 1 处选做
// 编译:  cd ~/AIInfraGuide/experiments && make saxpy
// 运行:  ./saxpy
// 验证:  程序自带 CPU 对照，输出 PASS 即正确
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// CUDA 调用错误检查：驱动失配/显存不足时立刻报错退出，避免"假 PASS"
#define CHECK(call) do {                                                    \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                            \
    }                                                                       \
} while (0)

#define N (1 << 24)     // 16M 个元素，x/y 各 64 MB，超出 L2
#define THREADS 256
#define REPS 10         // GPU 计时重复次数，取最优

// ------------------------------------------------------------
// TODO 1~2：SAXPY kernel
// ------------------------------------------------------------
__global__ void saxpy(const float* __restrict__ x, float* __restrict__ y, float a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO 1：越界判断。最后一个 block 里超出 n 的线程必须直接退出，
    //         否则会写坏别人（或别的分配）的内存。
    //         提示：if (idx >= n) return;
    if (idx >= n) return;


    // TODO 2：执行 SAXPY 本体，一个元素一次读改写。
    //         注意 y 在同一行既出现在等号左边也出现在右边。
    y[idx] = y[idx] + a * x[idx];


}

// ------------------------------------------------------------
// 选做 TODO 5：grid-stride 版本（对应 8.3 练习 3 的思路）
// 固定启动少量 block，每个线程用循环处理多个元素。
// 好处：grid 大小与 N 解耦，调优时只动 block 数。
// ------------------------------------------------------------
__global__ void saxpyStride(const float* __restrict__ x, float* __restrict__ y, float a, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // TODO 5：和 TODO 2 一样的表达式，只是索引换成循环变量 i


    }
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);
    const float a = 2.0f;

    // ---------- 主机端准备 ----------
    float* h_x = (float*)malloc(bytes);
    float* h_y = (float*)malloc(bytes);      // GPU 基准数据
    float* h_ref = (float*)malloc(bytes);    // CPU 对照数据
    for (int i = 0; i < N; i++) {
        h_x[i] = (i % 7) * 0.5f;             // 取值小，避免浮点误差放大
        h_y[i] = 1.0f;
        h_ref[i] = 1.0f;
    }

    // ---------- CPU 对照 + 计时 ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    // TODO 3：CPU 版 SAXPY，就地更新 h_ref
    for (int i = 0; i < N; i++) {
        h_ref[i] = a * h_x[i] + h_ref[i];
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ---------- 设备端 ----------
    float *d_x, *d_y;
    CHECK(cudaMalloc(&d_x, bytes));
    CHECK(cudaMalloc(&d_y, bytes));
    CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // TODO 4：计算 grid 并启动 saxpy（提示：向上取整，别漏掉尾部元素）
    //         int grid = (N + THREADS - 1) / THREADS;
    //         saxpy<<<grid, THREADS>>>(d_x, d_y, a, N);
    CHECK(cudaGetLastError());
    int grid = (N + THREADS - 1) / THREADS;
    saxpy<<<grid, THREADS>>>(d_x, d_y, a, N);
    CHECK(cudaGetLastError());


    CHECK(cudaDeviceSynchronize());

    // ---------- GPU 计时 ----------
    // 注意：SAXPY 是 in-place，每跑一次 y 都会变。计时循环里重复跑会让 y 越滚越大，
    // 但 kernel 的工作量与数据值无关，所以计时结果仍然有效；
    // 正确性校验前必须把 y 重置回初值，只跑一次。
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    saxpy<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_x, d_y, a, N);   // 预热
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    float bestMs = 1e9f;
    for (int k = 0; k < REPS; k++) {
        cudaEventRecord(e0);
        saxpy<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_x, d_y, a, N);
    CHECK(cudaGetLastError());
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms < bestMs) bestMs = ms;
    }

    // ---------- 正确性校验（重置 y，只跑一次） ----------
    CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));
    saxpy<<<(N + THREADS - 1) / THREADS, THREADS>>>(d_x, d_y, a, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    float maxDiff = 0.0f;
    for (int i = 0; i < N; i++) {
        float d = fabsf(h_y[i] - h_ref[i]);
        if (d > maxDiff) maxDiff = d;
    }

    // 每元素访存 12 字节（读 x + 读 y + 写 y），与 1.3 节 3.5 实验同一口径
    double gbps    = (double)N * 12 / (bestMs * 1e6);
    double cpuGbps = (double)N * 12 / (cpuMs  * 1e6);

    printf("SAXPY: N=%d, a=%.1f\n\n", N, a);
    printf("%-8s %12s %12s %10s %12s\n", "ver", "time(ms)", "items/ms", "speedup", "GB/s");
    printf("%-8s %12.3f %12.1f %10s %12.1f\n", "CPU", cpuMs, N / cpuMs, "1.0x", cpuGbps);
    printf("%-8s %12.3f %12.1f %9.1fx %12.1f\n", "GPU", bestMs, N / bestMs, cpuMs / bestMs, gbps);
    printf("\nmax|diff| = %.6f  %s\n", maxDiff, maxDiff < 1e-4f ? "PASS" : "FAIL");

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_ref);
    return 0;
}
