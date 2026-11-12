// ============================================================
// 练习 2.1：Warp 级操作观察台（warp_ops.cu）
//
// 只启动 <<<1, 32>>> —— 恰好一个 Warp，32 个 lane 全部参与，
// 每个 lane 把各条原语执行后「自己手里拿到的值」写进 out 的一行，
// host 端打印成表格并与 CPU 期望值逐位校验。
//
// 为什么这样设计：Warp Shuffle 最难的不是语法，而是脑子里想象
// 「我这个 lane 到底拿到了哪个 lane 的寄存器」。把 32 个 lane 的
// 真实结果摊在眼前，比读十遍文档有效。
//
// 规则速记：
//   laneId = threadIdx.x % 32（本例 <<<1,32>>>，所以 = threadIdx.x）
//   __shfl_sync(mask, var, srcLane)           源 lane = srcLane（段内编号）
//   __shfl_up_sync(mask, var, delta)          源 lane = laneId - delta
//   __shfl_down_sync(mask, var, delta)        源 lane = laneId + delta
//   __shfl_xor_sync(mask, var, laneMask)      源 lane = laneId ^ laneMask
//   越界 / 超出 width 分段时，保留自己的 var（不报错、不跨段）
//   mask=0xFFFFFFFF 表示全员参与；Volta+ 独立线程调度下必须显式给全掩码
//
// 编译:  cd ~/AIInfraGuide/experiments && make warp_ops
// 运行:  ./warp_ops
// ============================================================

#include <cstdio>
#include <cuda_runtime.h>

#define LANES 32
#define FULL  0xFFFFFFFFu

enum {
    R_LANE = 0, R_VAL, R_SHFL, R_UP, R_DOWN, R_XOR, R_WIDTH,
    R_BALLOT, R_ANY, R_ALL, R_REDUCE, R_XORREDUCE, R_ACTIVE, ROWS
};

static const char* rowName[ROWS] = {
    "lane", "val", "shfl", "up", "down", "xor", "width",
    "ballot", "any", "all", "reduce", "xorReduce", "activeMask"
};

#define CHECK(call) do {                                                    \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                __FILE__, __LINE__, cudaGetErrorString(_e));                 \
        return 1;                                                           \
    }                                                                       \
} while (0)

__global__ void warpProbeKernel(int* out) {
    int lane = threadIdx.x;      // <<<1,32>>> → lane 就是 threadIdx.x
    int val  = lane + 1;         // 1..32，便于心算期望值

    out[R_LANE * LANES + lane] = lane;
    out[R_VAL  * LANES + lane] = val;

    // TODO 1：广播 —— 所有 lane 读 lane 0 的 val，结果全为 1
    out[R_SHFL * LANES + lane] = __shfl_sync(FULL, val, 0);

    // TODO 2：三种相邻数据移动，delta / laneMask 都取 1
    out[R_UP   * LANES + lane] = __shfl_up_sync(FULL, val, 1);
    out[R_DOWN * LANES + lane] = __shfl_down_sync(FULL, val, 1);
    out[R_XOR  * LANES + lane] = __shfl_xor_sync(FULL, val, 1);
    // lane 0 的 up / lane 31 的 down 源 lane 越界 → 保留自己的 val（1 / 32），
    // 不回绕、不补零。xor 按编号配对，人人有伴，不会越界。

    // TODO 3：Warp Vote， pred = 本 lane 是否为偶数
    int pred = (lane % 2 == 0);
    out[R_BALLOT * LANES + lane] = (int)__ballot_sync(FULL, pred);
    out[R_ANY    * LANES + lane] = __any_sync(FULL, pred);
    out[R_ALL    * LANES + lane] = __all_sync(FULL, pred);
    // 偶数 lane 置位 → 位图 = 0x55555555；any=1，all=0。

    // TODO 4：width=8 逻辑分段广播，srcLane=0 指「本段段内 0」
    out[R_WIDTH * LANES + lane] = __shfl_sync(FULL, val, 0, 8);
    // lane 8 拿到的是 9 而不是 1：源 lane 不能跨段，
    // 但 mask 仍是全 32 —— width 只限制寻址，不改变参与集合。

    // TODO 5：两种 Warp 归约（val = lane+1，1..32 求和 = 528）
    int s = val;
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_down_sync(FULL, s, off);
    out[R_REDUCE * LANES + lane] = s;
    // 注意：down 版越界 lane 拿到的是「自己的 s」，累加等于自己加自己，
    // 5 步后 lane i = 528 + 16*i。只有 lane 0 是全和，其余是无效的
    // 自翻倍值 —— 这正是文档 6.1 只让 lane 0 写共享内存的原因。

    int x = val;
    for (int off = 16; off > 0; off >>= 1)
        x += __shfl_xor_sync(FULL, x, off);
    out[R_XORREDUCE * LANES + lane] = x;
    // xor 蝶形归约每步与配对 lane 互换累加，结果对称扩散 → 全员都是 528。

    // TODO 6：分支内观察活跃掩码（观察项，不参与 PASS 判定）
    unsigned am = FULL;
    if (lane < 16) am = __activemask();
    out[R_ACTIVE * LANES + lane] = (int)am;
    // 前 16 lane 通常得到 0x0000FFFF，其余 lane 保留 FULL；
    // 编译器可能提前 reconverge，所以这行只观察、不断言。
}

// CPU 期望值：每行每 lane 的正确答案（R_ACTIVE 行跳过校验）
int expect(int row, int lane) {
    int v = lane + 1;
    switch (row) {
        case R_LANE:      return lane;
        case R_VAL:       return v;
        case R_SHFL:      return 1;                       // 全拿到 lane0 的 val=1
        case R_UP:        return (lane == 0) ? 1 : lane; // 越界保留自己
        case R_DOWN:      return (lane == 31) ? 32 : lane + 2;
        case R_XOR:       return (lane ^ 1) + 1;
        case R_BALLOT:    return 0x55555555;             // 偶数 lane 置位
        case R_ANY:       return 1;
        case R_ALL:       return 0;
        case R_WIDTH:     return (lane / 8) * 8 + 1;     // 每段段首的 val
        case R_REDUCE:    return 528 + 16 * lane;        // down 归约越界自翻倍的结果
        case R_XORREDUCE: return 528;                    // xor 蝶形归约：全员全和
        default:          return 0;
    }
}

int main() {
    int h[ROWS * LANES];
    for (int i = 0; i < ROWS * LANES; i++) h[i] = -1;   // 哨兵：TODO 没填就是 FAIL

    int* d = nullptr;
    CHECK(cudaMalloc(&d, sizeof(h)));
    CHECK(cudaMemcpy(d, h, sizeof(h), cudaMemcpyHostToDevice));

    warpProbeKernel<<<1, LANES>>>(d);                    // 恰好一个 Warp
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    CHECK(cudaDeviceSynchronize());

    printf("Warp probe: block=1x32, one warp, val = lane+1 (1..32)\n\n");
    printf("--- data movement ---\n");
    printf("%4s %5s %6s %6s %6s %6s %7s\n",
           "lane", "val", "shfl", "up", "down", "xor", "width");
    for (int l = 0; l < LANES; l++)
        printf("%4d %5d %6d %6d %6d %6d %7d\n",
               h[R_LANE  * LANES + l], h[R_VAL   * LANES + l],
               h[R_SHFL  * LANES + l], h[R_UP    * LANES + l],
               h[R_DOWN  * LANES + l], h[R_XOR   * LANES + l],
               h[R_WIDTH * LANES + l]);

    printf("\n--- collective ops (ballot/activeMask in hex) ---\n");
    printf("%4s %10s %5s %5s %8s %10s %12s\n",
           "lane", "ballot", "any", "all", "reduce", "xorReduce", "activeMask");
    for (int l = 0; l < LANES; l++)
        printf("%4d %10x %5d %5d %8d %10d %12x\n",
               l, (unsigned)h[R_BALLOT * LANES + l],
               h[R_ANY     * LANES + l], h[R_ALL     * LANES + l],
               h[R_REDUCE  * LANES + l], h[R_XORREDUCE * LANES + l],
               (unsigned)h[R_ACTIVE * LANES + l]);

    int bad = 0, firstRow = -1, firstLane = -1, firstGot = 0, firstWant = 0;
    for (int r = 0; r < ROWS; r++) {
        if (r == R_ACTIVE) continue;                     // 观察项不判定
        for (int l = 0; l < LANES; l++) {
            int got = h[r * LANES + l], want = expect(r, l);
            if (got != want) {
                bad++;
                if (bad == 1) {
                    firstRow = r; firstLane = l; firstGot = got; firstWant = want;
                }
            }
        }
    }

    printf("\nresult: %s  (%d mismatches / %d checked)\n",
           bad ? "FAIL" : "PASS", bad, (ROWS - 1) * LANES);
    if (bad)
        printf("first mismatch: row=%s lane=%d got=%d want=%d%s\n",
               rowName[firstRow], firstLane, firstGot, firstWant,
               (firstGot == -1) ? "  <- sentinel, TODO not filled" : "");
    printf("\nnote: activeMask row is observation only (ITS may reconverge).\n");

    CHECK(cudaFree(d));
    return 0;
}
