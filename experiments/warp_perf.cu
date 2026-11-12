// ============================================================
// 练习 2.1：Warp 性能对比实验（warp_perf.cu）
//
// A. Divergence 代价（对应 2.1 文档 3.2 / 3.4）
//    三个 kernel 数学工作量相同，只改控制流分布：
//      divergent    : tid 奇偶分支 → 每 warp 16/16 分歧
//      warp-aligned : (tid>>5) 奇偶分支 → 同一 warp 走同一路径
//      branchless   : 算术选择替代分支，无分歧
//    CUDA Event 计时，重复 REPS 次取平均。
//
// B. Block 归约（对应 2.1 文档 6.2）
//    shared-only   : 纯共享内存折半归约，8+1 次 __syncthreads()
//    shuffle+shared: warp shuffle 归约 + 1 次 __syncthreads()
//    输入全为 1.0f → 每 block 精确得 256，便于逐位校验。
//
// 编译:  cd ~/AIInfraGuide/experiments && make warp_perf
// 运行:  ./warp_perf
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N       (1 << 24)   // 16M floats = 64MB，超过 L2，避免全命中缓存
#define THREADS 256
#define ITERS   200         // 每元素 200 次 FMA，让 kernel 时间远大于 launch 开销
#define REPS    10
#define FULL    0xFFFFFFFFu

#define CHECK(call) do {                                                    \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        exit(1);                                                            \
    }                                                                       \
} while (0)

// ---------- A. Divergence ----------
// __noinline__ 阻止编译器把两条路径合并成一次无条件循环（否则分歧会被优化掉）
__device__ __noinline__ float pathPlus(float v) {
    for (int k = 0; k < ITERS; k++) v = v * 1.000001f + 0.5f;
    return v;
}

__device__ __noinline__ float pathMinus(float v) {
    for (int k = 0; k < ITERS; k++) v = fmaf(v, 1.000001f, -0.5f);
    return v;
}

__global__ void divergentKernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if (tid & 1) out[tid] = pathMinus(in[tid]);   // 相邻 lane 路径不同
    else         out[tid] = pathPlus(in[tid]);
}

__global__ void alignedKernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if ((tid >> 5) & 1) out[tid] = pathMinus(in[tid]);  // 同一 warp 走同一路径
    else                out[tid] = pathPlus(in[tid]);
}

__global__ void branchlessKernel(const float* in, float* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float s = (tid & 1) ? -0.5f : 0.5f;      // 算术选择，无分支
    float v = in[tid];
    for (int k = 0; k < ITERS; k++) v = fmaf(v, 1.000001f, s);
    out[tid] = v;
}

// ---------- B. Block 归约 ----------
__device__ float warpReduceSum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(FULL, v, off);   // 只有 lane 0 的结果可信
    return v;
}

__global__ void reduceSharedKernel(const float* in, float* out, int n) {
    __shared__ float s[THREADS];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (tid < n) ? in[tid] : 0.0f;
    __syncthreads();
    for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}

__global__ void reduceShflKernel(const float* in, float* out, int n) {
    __shared__ float warpSums[THREADS / 32];   // 8 个 float = 32B
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    float v = (tid < n) ? in[tid] : 0.0f;
    v = warpReduceSum(v);
    if (lane == 0) warpSums[wid] = v;
    __syncthreads();                            // 只需 1 次块级同步

    float s = (threadIdx.x < THREADS / 32) ? warpSums[threadIdx.x] : 0.0f;
    if (wid == 0) s = warpReduceSum(s);         // 只有第一个 warp 参与
    if (threadIdx.x == 0) out[blockIdx.x] = s;
}

// ---------- 计时 ----------
template <typename F>
float timeKernel(F launch) {
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    launch();                                   // 预热（触发惰性加载、时钟稳定）
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(a));
    for (int r = 0; r < REPS; r++) launch();
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    CHECK(cudaEventDestroy(a));
    CHECK(cudaEventDestroy(b));
    return ms / REPS;
}

int main() {
    size_t bytes = (size_t)N * sizeof(float);
    int gridA = (N + THREADS - 1) / THREADS;         // 65536 blocks
    int blocksB = gridA;                              // 每 block 归约 256 个元素
    size_t outBytes = (size_t)blocksB * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_a  = (float*)malloc(bytes);
    float* h_b1 = (float*)malloc(outBytes);
    float* h_b2 = (float*)malloc(outBytes);
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;      // 全 1 → B 每 block 精确得 256

    float *d_in, *d_a, *d_b1, *d_b2;
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_a,  bytes));
    CHECK(cudaMalloc(&d_b1, outBytes));
    CHECK(cudaMalloc(&d_b2, outBytes));
    CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // ---- A ----
    float tDiv = timeKernel([&] { divergentKernel<<<gridA, THREADS>>>(d_in, d_a, N); });
    CHECK(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
    float tAli = timeKernel([&] { alignedKernel<<<gridA, THREADS>>>(d_in, d_a, N); });
    float tBrl = timeKernel([&] { branchlessKernel<<<gridA, THREADS>>>(d_in, d_a, N); });
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    // 抽样检查 A 的输出：三条路径数学不同，只验「都算过且数值有效」
    int badA = 0;
    for (int i = 0; i < N; i += 4096) {
        float v = h_a[i];
        if (!isfinite(v) || fabsf(v) > 1.0e4f || v == -999.0f) badA++;
    }

    // ---- B ----
    float tShr = timeKernel([&] { reduceSharedKernel<<<blocksB, THREADS>>>(d_in, d_b1, N); });
    float tShf = timeKernel([&] { reduceShflKernel<<<blocksB, THREADS>>>(d_in, d_b2, N); });
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h_b1, d_b1, outBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_b2, d_b2, outBytes, cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    double maxDiff = 0.0;
    for (int i = 0; i < blocksB; i++) {
        double d = fabs((double)h_b1[i] - (double)h_b2[i]);
        if (d > maxDiff) maxDiff = d;
    }

    printf("N=%d (%.0f MB)  THREADS=%d  ITERS=%d  REPS=%d  blocks=%d\n\n",
           N, (double)bytes / (1024 * 1024), THREADS, ITERS, REPS, gridA);

    printf("=== A. warp divergence (kernel time, ms) ===\n");
    printf("%-20s %12s %10s\n", "kernel", "time(ms)", "speedup");
    printf("%-20s %12.3f %9.2fx\n", "divergent(lane)",  tDiv, 1.0);
    printf("%-20s %12.3f %9.2fx\n", "warp-aligned",     tAli, tDiv / tAli);
    printf("%-20s %12.3f %9.2fx\n", "branchless",       tBrl, tDiv / tBrl);
    printf("sample check: %s (%d bad / %d samples)\n\n",
           badA ? "FAIL" : "PASS", badA, (N + 4095) / 4096);

    printf("=== B. block reduction (kernel time, ms) ===\n");
    printf("%-20s %12s %10s %14s\n", "kernel", "time(ms)", "speedup", "GB/s(read)");
    printf("%-20s %12.3f %9.2fx %14.1f\n",
           "shared-only", tShr, 1.0, (double)bytes / (tShr * 1e6));
    printf("%-20s %12.3f %9.2fx %14.1f\n",
           "shuffle+shared", tShf, tShr / tShf, (double)bytes / (tShf * 1e6));
    printf("maxDiff = %.6f  (expect 0: every block sums 256 x 1.0f)  %s\n",
           maxDiff, (maxDiff < 1e-4) ? "PASS" : "FAIL");

    CHECK(cudaFree(d_in)); CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b1)); CHECK(cudaFree(d_b2));
    free(h_in); free(h_a); free(h_b1); free(h_b2);
    return 0;
}
