// ============================================================
// 练习 8.3：Grid-stride Loop 版本 + 启动规模扫描
//
// 和 8.1/8.2 的"精确覆盖"（grid = ceil(N/THREADS)，一线程一个元素）不同，
// grid-stride 固定启动若干 block，每个线程用循环处理多个元素：
//     for (int i = idx; i < n; i += totalThreads) ...
//
// 本题的两个看点：
//   1) N 故意不是 256 的整数倍（见 #define N），grid-stride 天然不需要向上取整
//   2) 扫描 block 数（32 / 64 / 128 / 256 / 1024 / 4096），看"启动多少 block 最合适"
//
// 你的任务：补全 4 处 TODO + 1 处选做
// 编译:  cd ~/AIInfraGuide/experiments && make vectorAddStride
// 运行:  ./vectorAddStride
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N ((1 << 24) + 137)   // 故意留个尾巴：不是 256 的整数倍
#define THREADS 256
#define REPS 5

// CUDA 调用错误检查（8.2 那次"假 PASS"的教训）
#define CHECK(call) do {                                                    \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                            \
    }                                                                       \
} while (0)

// ---------- 基准：精确覆盖版，每个线程处理一个元素 ----------
__global__ void vectorAddExact(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

// ------------------------------------------------------------
// TODO 1~2：grid-stride 版本
// ------------------------------------------------------------
__global__ void vectorAddStride(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO 1：算出"每轮跳跃的步长" total（= 一个 grid 里总共有多少线程）
    //         提示：blockDim.x * gridDim.x
    int stride = blockDim.x *gridDim.x;


    // TODO 2：grid-stride 循环。每个线程从自己的 idx 出发，
    //         每次跨 total 个元素，直到越界为止；循环体内做 C[i] = A[i] + B[i]。
    //         注意：循环条件本身就完成了边界检查，所以不需要额外的 if。
    for (int i = idx; i<n; i+=stride){
        C[i] = A[i] + B[i];
    }


}

// 计时：预热 1 次 + 跑 REPS 次取最优
template <typename F>
float bench(F&& launch) {
    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    launch();
    CHECK(cudaDeviceSynchronize());

    float best = 1e9f;
    for (int k = 0; k < REPS; k++) {
        CHECK(cudaEventRecord(t0));
        launch();
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, t0, t1));
        if (ms < best) best = ms;
    }
    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return best;
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = (i % 17) * 0.25f;
        h_B[i] = (i % 5) * 0.5f;
        h_ref[i] = h_A[i] + h_B[i];       // CPU 对照真值
    }

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_B, bytes));
    CHECK(cudaMalloc(&d_C, bytes));
    CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // ---------- 基准：精确覆盖 ----------
    int exactGrid = (N + THREADS - 1) / THREADS;
    float msExact = bench([&]{
        vectorAddExact<<<exactGrid, THREADS>>>(d_A, d_B, d_C, N);
        CHECK(cudaGetLastError());
    });

    // ---------- 扫描：固定 THREADS，只改 block 数 ----------
    int gridSizes[] = {32, 64, 128, 256, 1024, 4096};
    const int count = sizeof(gridSizes) / sizeof(gridSizes[0]);
    double bytesTotal = (double)N * 12;   // 读 A + 读 B + 写 C

    printf("N = %d (not a multiple of %d), THREADS = %d\n\n", N, THREADS, THREADS);
    printf("%-10s %12s %12s %12s %12s %10s\n",
           "blocks", "time(ms)", "GB/s", "vs exact", "maxDiff", "status");

    for (int k = 0; k < count; k++) {
        int g = gridSizes[k];

        // TODO 3：用当前扫描到的 block 数 g 启动 grid-stride kernel
        //         提示：vectorAddStride<<<g, THREADS>>>(d_A, d_B, d_C, N);
        //         别忘了紧跟一行 CHECK(cudaGetLastError());
        vectorAddStride<<<g, THREADS>>>(d_A, d_B, d_C, N);
        CHECK(cudaGetLastError());

        CHECK(cudaDeviceSynchronize());

        float ms = bench([&]{
            // TODO 3 同上：这里也要用 g 启动一次（bench 会重复调用这个 lambda）
            vectorAddStride<<<g, THREADS>>>(d_A, d_B, d_C, N);
            CHECK(cudaGetLastError());
        });

        CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        // TODO 4：逐元素比对 h_C 与 h_ref，求最大绝对误差
        //         初值故意给 -1：如果你没真的算，它会一直显示 -1.000000，
        //         避免像 8.2 那次一样出现"空循环 = diff 0 = 假 PASS"
        float maxDiff = -1.0f;
        for (int i = 0; i < N; i++) {           // ① 遍历全部 16777353 个元素
            float d = fabsf(h_C[i] - h_ref[i]); // ② 算第 i 个位置上 GPU 和 CPU 的偏差
            if (d > maxDiff) maxDiff = d;       // ③ 记录到目前为止见过的最大偏差
        }



        printf("%-10d %12.3f %12.1f %11.2fx %12.6f %10s\n",
               g, ms, bytesTotal / (ms * 1e6), msExact / ms, maxDiff,
               (maxDiff >= 0.0f && maxDiff < 1e-4f) ? "PASS" : "FAIL");
    }

    printf("\n%-10s %12.3f %12.1f %11s %12s\n",
           "exact", msExact, bytesTotal / (msExact * 1e6), "1.00x", "-");

    // 思考（不用写代码）：
    //  - block 数太小：每个线程要循环很多次，指令/分支开销占比上升
    //  - block 数太大：绝大多数 block 只跑一轮就结束，调度开销占比上升
    //  - 最优点通常靠近 "SM 数 × 每 SM 可驻留 block 数"（这叫充分占用，见 1.2 节）
    //  - 选做 TODO 5：把 N 改成一个能整除的值，看结果曲线有没有变化

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
