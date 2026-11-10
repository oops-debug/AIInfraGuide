// ============================================================
// 练习 8.1：向量点积  result = sum(A[i] * B[i])
//
// 思路（两阶段规约）：
//   阶段一：每个 Block 内，线程各自算乘积项 -> 共享内存 -> 树形规约
//   阶段二：每个 Block 的部分和 用 atomicAdd 汇总到全局结果
//
// 你的任务：补全 5 处 TODO（kernel 3 处 + main 2 处）
// 编译:  nvcc -O3 -arch=sm_120 dotProduct.cu -o dotProduct
// 验证:  程序自带 CPU 对照，输出 PASS 即正确
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define N (1 << 20)   // 1M 个元素
#define THREADS 256   // 每个 Block 的线程数

// ------------------------------------------------------------
// TODO 1~3：规约 kernel
// ------------------------------------------------------------
__global__ void dotProduct(const float* A, const float* B, float* result, int n) {
    // 块内共享内存：每个线程贡献一个部分和
    __shared__ float sdata[THREADS];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO 1：把 A[idx]*B[idx] 写入 sdata[tid]
    // 注意：idx 可能越界（N 不是 THREADS 的整数倍时），越界的线程要写 0
    // 提示：sdata[tid] = (idx < n) ? ... : ...;
    sdata[tid] = (idx < n) ? A[idx]*B[idx]:0.0f;


    __syncthreads();   // 等 block 内所有线程都写完共享内存

    // TODO 2：树形规约
    // 思路：每轮让一半线程把"自己 + 对面镜像位置"加起来，活动线程数每轮减半
    // 例如 8 个元素：[a0 a1 a2 a3 | a4 a5 a6 a7] -> 前4个线程各做一次加 -> [s0 s1 s2 s3] -> ...
    // 提示：for (int s = blockDim.x / 2; s > 0; s >>= 1)
    //       循环体内：if (tid < s) sdata[tid] += sdata[tid + s];
    //       每轮加了共享内存之后都要 __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s>>=1){
        if (tid < s) sdata[tid] += sdata[tid+s];
        __syncthreads();
    }



    // TODO 3：块内规约结束后，和值在 sdata[0]。
    // 让 0 号线程用 atomicAdd 把它累加到全局的 result 上
    // （多个 block 并发执行，必须原子操作，否则互相覆盖）
    if (tid == 0)
        atomicAdd(result, sdata[0]);



}

int main() {
    // ---------- 主机端分配与初始化 ----------
    size_t bytes = N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;                  // 故意用简单值，方便口算验证：点积应 = N
        h_B[i] = 1.0f;
    }

    // ---------- 设备端分配 ----------
    float *d_A, *d_B, *d_result;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_result, sizeof(float));

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(float));   // 结果从 0 开始累加

    // TODO 4：计算并启动 kernel
    // 提示：block 数 = (N + THREADS - 1) / THREADS，正好覆盖所有元素
    //       grid 变量声明在这行上面自己写：int grid = ...;
    int grid =( N + THREADS - 1)/THREADS;
    dotProduct<<< grid, THREADS>>>(d_A, d_B, d_result, N);
    cudaDeviceSynchronize();

    // ---------- 取回结果并校验 ----------
    float gpuResult;
    cudaMemcpy(&gpuResult, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // TODO 5：CPU 端自己算一遍对照（一个 for 循环）
    float cpuResult = 0.0f;
    for (int i = 0; i < N; i++) {
        cpuResult += h_A[i] * h_B[i];
    }


    printf("GPU result: %.1f\n", gpuResult);
    printf("CPU result: %.1f\n", cpuResult);
    printf("%s (误差 %.6f)\n",
           fabsf(gpuResult - cpuResult) < 1e-2f ? "PASS" : "FAIL",
           fabsf(gpuResult - cpuResult));

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_result);
    free(h_A); free(h_B);
    return 0;
}
