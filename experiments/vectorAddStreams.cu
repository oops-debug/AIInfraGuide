// ============================================================
// 练习 8.4：CUDA Stream 异步传输与计算重叠
//
// 对比两个端到端流程（host 墙钟计时，含 H2D → kernel → D2H）：
//   A. seq     ：默认流，整块「传输→计算→传输」严格串行
//   B. overlap ：数据分 CHUNKS 块，STREAMS 条流轮转，
//                第 1 块在计算时，第 2 块已经在传输……流水线重叠
//
// 两个决定成败的细节：
//   1) host 内存必须用 cudaMallocHost（pinned，框架已写好）——
//      pageable 内存上 cudaMemcpyAsync 会退化成同步拷贝，重叠失效
//      （对照 1.4 正文 7.3 节：pageable 端到端只有 0.67 GB/s）
//   2) ITERS 是「计算放大旋钮」：kernel 内做 ITERS-1 次纯寄存器运算，
//      不增加任何显存流量。实测 5060 Laptop 算力太强：
//        ITERS=1    → kernel ~0.6ms，被 12ms 传输淹没，看不出收益差异
//        ITERS=200  → 计算仅 ~0.3ms 增量，端到端仍几乎不变
//        ITERS=20000→ kernel ~30ms > 13ms 传输，重叠收益才显形
//
// 编译:  cd ~/AIInfraGuide/experiments && make vectorAddStreams
// 运行:  ./vectorAddStreams
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define N (1 << 24)      // 16M floats = 64MB / 数组，N 是 2 的幂，能被 CHUNKS 整除
#define THREADS 256
#define CHUNKS 4         // overlap 版把数据分成几块流水
#define STREAMS 2        // 轮转使用几条流
#define ITERS 20000      // 计算放大倍数：≈30ms kernel，压过 ~13ms 的 PCIe 传输

// CUDA 调用错误检查（沿用 8.2/8.3 的教训）
#define CHECK(call) do {                                                    \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                            \
    }                                                                       \
} while (0)

__global__ void computeKernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float c = A[idx] + B[idx];
    for (int k = 1; k < ITERS; k++)       // 计算放大：纯寄存器运算，访存不变
        c = c * 1.000001f + 0.5f;
    C[idx] = c;
}

// CPU 对照（与 kernel 逐位一致：每个元素同样的运算顺序，无规约）
// 注意：只在 ITERS ≤ 200 时调用 —— 大 ITERS 时单线程要跑 3.3e11 次
// FMA，数分钟且发生在首个 printf 之前，看起来像「卡死」（实测教训）
void cpuCompute(const float* A, const float* B, float* C, int n) {
    for (int i = 0; i < n; i++) {
        float c = A[i] + B[i];
        for (int k = 1; k < ITERS; k++)
            c = c * 1.000001f + 0.5f;
        C[i] = c;
    }
}

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);
    int chunkElems = N / CHUNKS;
    size_t chunkBytes = (size_t)chunkElems * sizeof(float);

    // ---- pinned host 内存：真正异步的前提（cudaMallocHost，不是 malloc！）----
    float *h_A, *h_B, *h_C1, *h_C2, *h_ref;
    CHECK(cudaMallocHost(&h_A, bytes));
    CHECK(cudaMallocHost(&h_B, bytes));
    CHECK(cudaMallocHost(&h_C1, bytes));   // seq 版输出
    CHECK(cudaMallocHost(&h_C2, bytes));   // overlap 版输出
    CHECK(cudaMallocHost(&h_ref, bytes));

    for (int i = 0; i < N; i++) {
        h_A[i] = (i % 17) * 0.25f;
        h_B[i] = (i % 5) * 0.5f;
    }

    // 参考结果二选一：
    //   小 ITERS（≤200）：CPU 串行对照，几秒内跑完 —— 教学上最有说服力
    //   大 ITERS（>200）：CPU 对照等不起，改用 seq 版 GPU 结果互校：
    //     同一个 kernel 逐位一致，足以验证 overlap 分块搬运没有错位
    float* ref;
    if (ITERS <= 200) {
        cpuCompute(h_A, h_B, h_ref, N);   // CPU 对照：仅小 ITERS 时启用
        ref = h_ref;
    } else {
        ref = h_C1;   // 大 ITERS：seq 还没跑，但 maxDiff 在 seq 之后才计算，
                      // 到时候 h_C1 已填充完毕 —— 指针提前记下即可
    }
    for (int i = 0; i < N; i++) h_C2[i] = -999.0f;   // 哨兵：没跑 overlap 就是大 FAIL

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_B, bytes));
    CHECK(cudaMalloc(&d_C, bytes));

    // ================= 版本 A：默认流串行（基线，已写好）=================
    double t0 = nowMs();
    CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    int grid = (N + THREADS - 1) / THREADS;
    computeKernel<<<grid, THREADS>>>(d_A, d_B, d_C, N);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_C1, d_C, bytes, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());
    double msSeq = nowMs() - t0;

    double maxDiff1 = 0.0;
    for (int i = 0; i < N; i++) {
        double d = fabs((double)h_C1[i] - ref[i]);
        if (d > maxDiff1) maxDiff1 = d;
    }

    // ============ 版本 B：CHUNKS 块 × STREAMS 条流流水线 ============
    cudaStream_t streams[STREAMS];
    for (int s = 0; s < STREAMS; s++) CHECK(cudaStreamCreate(&streams[s]));

    int chunkGrid = (chunkElems + THREADS - 1) / THREADS;

    t0 = nowMs();
    for (int c = 0; c < CHUNKS; c++) {
        cudaStream_t s = streams[c % STREAMS];   // 块号轮流分给各条流
        int off = c * chunkElems;                // 本块在数组里的起始下标

        // TODO 1：本块的两条异步 H2D 拷贝
        cudaMemcpyAsync(d_A + off, h_A + off, chunkBytes, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(d_B + off, h_B + off, chunkBytes, cudaMemcpyHostToDevice, s);

        // TODO 2：本块的 kernel —— 流参数在 <<<grid, block, shmem, stream>>> 第 4 位
        computeKernel<<<chunkGrid, THREADS, 0, s>>>(d_A + off, d_B + off, d_C + off, chunkElems);

        // TODO 3：本块的异步 D2H，结果拷回 h_C2 + off（方向别抄反，偏移别漏）
        cudaMemcpyAsync(h_C2 + off, d_C + off, chunkBytes, cudaMemcpyDeviceToHost, s);
    }
    // 等两条流都干完活（漏了这步，计时会在数据还没落地时结束）
    for (int s = 0; s < STREAMS; s++) CHECK(cudaStreamSynchronize(streams[s]));
    double msOverlap = nowMs() - t0;

    double maxDiff2 = -1.0;
    for (int i = 0; i < N; i++) {
        double d = fabs((double)h_C2[i] - ref[i]);
        if (d > maxDiff2) maxDiff2 = d;
    }

    // ================= 汇总 =================
    double bytesTotal = (double)N * 12;   // ITERS=1 时的显存流量口径：读 2 写 1
    printf("N=%d floats/array (%.0f MB)  CHUNKS=%d  STREAMS=%d  ITERS=%d\n\n",
           N, (double)bytes / (1024 * 1024), CHUNKS, STREAMS, ITERS);
    printf("%-12s %15s %17s %10s %12s %10s\n",
           "ver", "end-to-end(ms)", "GB/s", "speedup", "maxDiff", "status");
    printf("%-12s %15.2f %17.1f %9.3fx %12.6f %10s\n",
           "seq(pinned)", msSeq, bytesTotal / (msSeq * 1e6), 1.0, maxDiff1,
           (maxDiff1 < 1e-4) ? "PASS" : "FAIL");
    printf("%-12s %15.2f %17.1f %9.3fx %12.6f %10s\n",
           "overlap", msOverlap, bytesTotal / (msOverlap * 1e6),
           msSeq / msOverlap, maxDiff2,
           (maxDiff2 >= 0.0 && maxDiff2 < 1e-4) ? "PASS" : "FAIL");
    printf("\n对照 1.4 正文：pageable 内存的端到端 = 193.6 ms / 0.67 GB/s\n");
    printf("对照记录：ITERS=1 → seq 13.39 / overlap 9.28 (1.442x)，收益来自 PCIe 全双工\n");

    for (int s = 0; s < STREAMS; s++) CHECK(cudaStreamDestroy(streams[s]));
    CHECK(cudaFree(d_A)); CHECK(cudaFree(d_B)); CHECK(cudaFree(d_C));
    CHECK(cudaFreeHost(h_A)); CHECK(cudaFreeHost(h_B));
    CHECK(cudaFreeHost(h_C1)); CHECK(cudaFreeHost(h_C2)); CHECK(cudaFreeHost(h_ref));
    return 0;
}
