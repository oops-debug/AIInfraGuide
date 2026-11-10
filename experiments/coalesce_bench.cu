#include <cstdio>
#include <cuda_runtime.h>

#define N (256 * 1024 * 1024)  // 256M floats = 1 GiB，远大于 L2（几十 MB），保证读操作穿透到显存

// ✅ 合并访问：warp 内连续线程读连续地址，每个元素只读一次
__global__ void readCoalesced(const float* __restrict__ in, float* out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    float s = 0.0f;
    for (int i = gid; i < n; i += total)            // grid-stride loop
        s += in[i];
    if (s == 12345.678f) out[gid] = s;              // 防止死代码消除
}

// ❌ 跨步访问：warp 内线程 i 读 in[i * STRIDE]，每个元素只读一次
template <int STRIDE>
__global__ void readStrided(const float* __restrict__ in, float* out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    float s = 0.0f;
    for (int i = gid; i < n / STRIDE; i += total)
        s += in[i * STRIDE];
    if (s == 12345.678f) out[gid] = s;
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
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // CUDA 13 移除了 prop.memoryClockRate / prop.memoryBusWidth，
    // 改用 cudaDeviceGetAttribute 兼容 12.x / 13.x
    int memClockKHz = 0, busWidthBits = 0;
    cudaDeviceGetAttribute(&memClockKHz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&busWidthBits, cudaDevAttrGlobalMemoryBusWidth, 0);
    // 注意：GDDR6/7 的实际数据率远高于 clock*2，此值仅为粗略参考
    double theoreticalGBps = memClockKHz > 0
        ? memClockKHz * 1e3 * 2.0 * (busWidthBits / 8.0) / 1e9 : 0.0;

    printf("GPU: %s | %d MB | theoretical ~%.0f GB/s\n\n",
           prop.name, (int)(prop.totalGlobalMem >> 20), theoreticalGBps);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, 1024 * 256 * sizeof(float));
    cudaMemset(d_in, 0, N * sizeof(float));

    dim3 grid(1024), block(256);

    // 各模式读的"有效字节数"不同：stride=1 读满 1 GiB，stride=2 读 512 MB，stride=32 只读 32 MB
    long long useful1  = (long long)N * 4;
    long long useful2  = (long long)N / 2 * 4;
    long long useful32 = (long long)N / 32 * 4;

    float ms1  = bench([&]{ readCoalesced<<<grid, block>>>(d_in, d_out, N); });
    float ms2  = bench([&]{ readStrided<2><<<grid, block>>>(d_in, d_out, N); });
    float ms32 = bench([&]{ readStrided<32><<<grid, block>>>(d_in, d_out, N); });

    printf("%-12s %10s %14s\n", "模式", "耗时(ms)", "有效带宽");
    printf("%-12s %10.3f %11.1f GB/s\n", "stride=1",  ms1,  useful1  / ms1  / 1e6);
    printf("%-12s %10.3f %11.1f GB/s\n", "stride=2",  ms2,  useful2  / ms2  / 1e6);
    printf("%-12s %10.3f %11.1f GB/s\n", "stride=32", ms32, useful32 / ms32 / 1e6);

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}