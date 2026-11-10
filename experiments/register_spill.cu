
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>

#define CHECK(call) do {                                      \
    cudaError_t e = (call);                                  \
    if (e != cudaSuccess) {                                  \
        fprintf(stderr, "%s\n", cudaGetErrorString(e));      \
        std::exit(1);                                        \
    }                                                        \
} while (0)

constexpr int N = 1 << 16;
constexpr int BLOCK = 256;

__global__ void spill_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 每个线程创建 96 个独立的 float 变量
    float r0 = input[(tid + 0 * BLOCK) & (N - 1)];
    float r1 = input[(tid + 1 * BLOCK) & (N - 1)];
    float r2 = input[(tid + 2 * BLOCK) & (N - 1)];
    float r3 = input[(tid + 3 * BLOCK) & (N - 1)];
    float r4 = input[(tid + 4 * BLOCK) & (N - 1)];
    float r5 = input[(tid + 5 * BLOCK) & (N - 1)];
    float r6 = input[(tid + 6 * BLOCK) & (N - 1)];
    float r7 = input[(tid + 7 * BLOCK) & (N - 1)];
    float r8 = input[(tid + 8 * BLOCK) & (N - 1)];
    float r9 = input[(tid + 9 * BLOCK) & (N - 1)];
    float r10 = input[(tid + 10 * BLOCK) & (N - 1)];
    float r11 = input[(tid + 11 * BLOCK) & (N - 1)];
    float r12 = input[(tid + 12 * BLOCK) & (N - 1)];
    float r13 = input[(tid + 13 * BLOCK) & (N - 1)];
    float r14 = input[(tid + 14 * BLOCK) & (N - 1)];
    float r15 = input[(tid + 15 * BLOCK) & (N - 1)];
    float r16 = input[(tid + 16 * BLOCK) & (N - 1)];
    float r17 = input[(tid + 17 * BLOCK) & (N - 1)];
    float r18 = input[(tid + 18 * BLOCK) & (N - 1)];
    float r19 = input[(tid + 19 * BLOCK) & (N - 1)];
    float r20 = input[(tid + 20 * BLOCK) & (N - 1)];
    float r21 = input[(tid + 21 * BLOCK) & (N - 1)];
    float r22 = input[(tid + 22 * BLOCK) & (N - 1)];
    float r23 = input[(tid + 23 * BLOCK) & (N - 1)];
    float r24 = input[(tid + 24 * BLOCK) & (N - 1)];
    float r25 = input[(tid + 25 * BLOCK) & (N - 1)];
    float r26 = input[(tid + 26 * BLOCK) & (N - 1)];
    float r27 = input[(tid + 27 * BLOCK) & (N - 1)];
    float r28 = input[(tid + 28 * BLOCK) & (N - 1)];
    float r29 = input[(tid + 29 * BLOCK) & (N - 1)];
    float r30 = input[(tid + 30 * BLOCK) & (N - 1)];
    float r31 = input[(tid + 31 * BLOCK) & (N - 1)];
    float r32 = input[(tid + 32 * BLOCK) & (N - 1)];
    float r33 = input[(tid + 33 * BLOCK) & (N - 1)];
    float r34 = input[(tid + 34 * BLOCK) & (N - 1)];
    float r35 = input[(tid + 35 * BLOCK) & (N - 1)];
    float r36 = input[(tid + 36 * BLOCK) & (N - 1)];
    float r37 = input[(tid + 37 * BLOCK) & (N - 1)];
    float r38 = input[(tid + 38 * BLOCK) & (N - 1)];
    float r39 = input[(tid + 39 * BLOCK) & (N - 1)];
    float r40 = input[(tid + 40 * BLOCK) & (N - 1)];
    float r41 = input[(tid + 41 * BLOCK) & (N - 1)];
    float r42 = input[(tid + 42 * BLOCK) & (N - 1)];
    float r43 = input[(tid + 43 * BLOCK) & (N - 1)];
    float r44 = input[(tid + 44 * BLOCK) & (N - 1)];
    float r45 = input[(tid + 45 * BLOCK) & (N - 1)];
    float r46 = input[(tid + 46 * BLOCK) & (N - 1)];
    float r47 = input[(tid + 47 * BLOCK) & (N - 1)];
    float r48 = input[(tid + 48 * BLOCK) & (N - 1)];
    float r49 = input[(tid + 49 * BLOCK) & (N - 1)];
    float r50 = input[(tid + 50 * BLOCK) & (N - 1)];
    float r51 = input[(tid + 51 * BLOCK) & (N - 1)];
    float r52 = input[(tid + 52 * BLOCK) & (N - 1)];
    float r53 = input[(tid + 53 * BLOCK) & (N - 1)];
    float r54 = input[(tid + 54 * BLOCK) & (N - 1)];
    float r55 = input[(tid + 55 * BLOCK) & (N - 1)];
    float r56 = input[(tid + 56 * BLOCK) & (N - 1)];
    float r57 = input[(tid + 57 * BLOCK) & (N - 1)];
    float r58 = input[(tid + 58 * BLOCK) & (N - 1)];
    float r59 = input[(tid + 59 * BLOCK) & (N - 1)];
    float r60 = input[(tid + 60 * BLOCK) & (N - 1)];
    float r61 = input[(tid + 61 * BLOCK) & (N - 1)];
    float r62 = input[(tid + 62 * BLOCK) & (N - 1)];
    float r63 = input[(tid + 63 * BLOCK) & (N - 1)];
    float r64 = input[(tid + 64 * BLOCK) & (N - 1)];
    float r65 = input[(tid + 65 * BLOCK) & (N - 1)];
    float r66 = input[(tid + 66 * BLOCK) & (N - 1)];
    float r67 = input[(tid + 67 * BLOCK) & (N - 1)];
    float r68 = input[(tid + 68 * BLOCK) & (N - 1)];
    float r69 = input[(tid + 69 * BLOCK) & (N - 1)];
    float r70 = input[(tid + 70 * BLOCK) & (N - 1)];
    float r71 = input[(tid + 71 * BLOCK) & (N - 1)];
    float r72 = input[(tid + 72 * BLOCK) & (N - 1)];
    float r73 = input[(tid + 73 * BLOCK) & (N - 1)];
    float r74 = input[(tid + 74 * BLOCK) & (N - 1)];
    float r75 = input[(tid + 75 * BLOCK) & (N - 1)];
    float r76 = input[(tid + 76 * BLOCK) & (N - 1)];
    float r77 = input[(tid + 77 * BLOCK) & (N - 1)];
    float r78 = input[(tid + 78 * BLOCK) & (N - 1)];
    float r79 = input[(tid + 79 * BLOCK) & (N - 1)];
    float r80 = input[(tid + 80 * BLOCK) & (N - 1)];
    float r81 = input[(tid + 81 * BLOCK) & (N - 1)];
    float r82 = input[(tid + 82 * BLOCK) & (N - 1)];
    float r83 = input[(tid + 83 * BLOCK) & (N - 1)];
    float r84 = input[(tid + 84 * BLOCK) & (N - 1)];
    float r85 = input[(tid + 85 * BLOCK) & (N - 1)];
    float r86 = input[(tid + 86 * BLOCK) & (N - 1)];
    float r87 = input[(tid + 87 * BLOCK) & (N - 1)];
    float r88 = input[(tid + 88 * BLOCK) & (N - 1)];
    float r89 = input[(tid + 89 * BLOCK) & (N - 1)];
    float r90 = input[(tid + 90 * BLOCK) & (N - 1)];
    float r91 = input[(tid + 91 * BLOCK) & (N - 1)];
    float r92 = input[(tid + 92 * BLOCK) & (N - 1)];
    float r93 = input[(tid + 93 * BLOCK) & (N - 1)];
    float r94 = input[(tid + 94 * BLOCK) & (N - 1)];
    float r95 = input[(tid + 95 * BLOCK) & (N - 1)];

    // 尽量让所有变量在循环中保持存活
    for (int j = 0; j < iters; ++j) {
        r0 = fmaf(r0, 0.9990000f, 0.0001000f);
        r1 = fmaf(r1, 0.9990010f, 0.0002000f);
        r2 = fmaf(r2, 0.9990020f, 0.0003000f);
        r3 = fmaf(r3, 0.9990030f, 0.0004000f);
        r4 = fmaf(r4, 0.9990040f, 0.0005000f);
        r5 = fmaf(r5, 0.9990050f, 0.0006000f);
        r6 = fmaf(r6, 0.9990060f, 0.0007000f);
        r7 = fmaf(r7, 0.9990070f, 0.0008000f);
        r8 = fmaf(r8, 0.9990080f, 0.0009000f);
        r9 = fmaf(r9, 0.9990090f, 0.0010000f);
        r10 = fmaf(r10, 0.9990100f, 0.0011000f);
        r11 = fmaf(r11, 0.9990110f, 0.0012000f);
        r12 = fmaf(r12, 0.9990120f, 0.0013000f);
        r13 = fmaf(r13, 0.9990130f, 0.0014000f);
        r14 = fmaf(r14, 0.9990140f, 0.0015000f);
        r15 = fmaf(r15, 0.9990150f, 0.0016000f);
        r16 = fmaf(r16, 0.9990160f, 0.0017000f);
        r17 = fmaf(r17, 0.9990170f, 0.0018000f);
        r18 = fmaf(r18, 0.9990180f, 0.0019000f);
        r19 = fmaf(r19, 0.9990190f, 0.0020000f);
        r20 = fmaf(r20, 0.9990200f, 0.0021000f);
        r21 = fmaf(r21, 0.9990210f, 0.0022000f);
        r22 = fmaf(r22, 0.9990220f, 0.0023000f);
        r23 = fmaf(r23, 0.9990230f, 0.0024000f);
        r24 = fmaf(r24, 0.9990240f, 0.0025000f);
        r25 = fmaf(r25, 0.9990250f, 0.0026000f);
        r26 = fmaf(r26, 0.9990260f, 0.0027000f);
        r27 = fmaf(r27, 0.9990270f, 0.0028000f);
        r28 = fmaf(r28, 0.9990280f, 0.0029000f);
        r29 = fmaf(r29, 0.9990290f, 0.0030000f);
        r30 = fmaf(r30, 0.9990300f, 0.0031000f);
        r31 = fmaf(r31, 0.9990310f, 0.0032000f);
        r32 = fmaf(r32, 0.9990320f, 0.0033000f);
        r33 = fmaf(r33, 0.9990330f, 0.0034000f);
        r34 = fmaf(r34, 0.9990340f, 0.0035000f);
        r35 = fmaf(r35, 0.9990350f, 0.0036000f);
        r36 = fmaf(r36, 0.9990360f, 0.0037000f);
        r37 = fmaf(r37, 0.9990370f, 0.0038000f);
        r38 = fmaf(r38, 0.9990380f, 0.0039000f);
        r39 = fmaf(r39, 0.9990390f, 0.0040000f);
        r40 = fmaf(r40, 0.9990400f, 0.0041000f);
        r41 = fmaf(r41, 0.9990410f, 0.0042000f);
        r42 = fmaf(r42, 0.9990420f, 0.0043000f);
        r43 = fmaf(r43, 0.9990430f, 0.0044000f);
        r44 = fmaf(r44, 0.9990440f, 0.0045000f);
        r45 = fmaf(r45, 0.9990450f, 0.0046000f);
        r46 = fmaf(r46, 0.9990460f, 0.0047000f);
        r47 = fmaf(r47, 0.9990470f, 0.0048000f);
        r48 = fmaf(r48, 0.9990480f, 0.0049000f);
        r49 = fmaf(r49, 0.9990490f, 0.0050000f);
        r50 = fmaf(r50, 0.9990500f, 0.0051000f);
        r51 = fmaf(r51, 0.9990510f, 0.0052000f);
        r52 = fmaf(r52, 0.9990520f, 0.0053000f);
        r53 = fmaf(r53, 0.9990530f, 0.0054000f);
        r54 = fmaf(r54, 0.9990540f, 0.0055000f);
        r55 = fmaf(r55, 0.9990550f, 0.0056000f);
        r56 = fmaf(r56, 0.9990560f, 0.0057000f);
        r57 = fmaf(r57, 0.9990570f, 0.0058000f);
        r58 = fmaf(r58, 0.9990580f, 0.0059000f);
        r59 = fmaf(r59, 0.9990590f, 0.0060000f);
        r60 = fmaf(r60, 0.9990600f, 0.0061000f);
        r61 = fmaf(r61, 0.9990610f, 0.0062000f);
        r62 = fmaf(r62, 0.9990620f, 0.0063000f);
        r63 = fmaf(r63, 0.9990630f, 0.0064000f);
        r64 = fmaf(r64, 0.9990640f, 0.0065000f);
        r65 = fmaf(r65, 0.9990650f, 0.0066000f);
        r66 = fmaf(r66, 0.9990660f, 0.0067000f);
        r67 = fmaf(r67, 0.9990670f, 0.0068000f);
        r68 = fmaf(r68, 0.9990680f, 0.0069000f);
        r69 = fmaf(r69, 0.9990690f, 0.0070000f);
        r70 = fmaf(r70, 0.9990700f, 0.0071000f);
        r71 = fmaf(r71, 0.9990710f, 0.0072000f);
        r72 = fmaf(r72, 0.9990720f, 0.0073000f);
        r73 = fmaf(r73, 0.9990730f, 0.0074000f);
        r74 = fmaf(r74, 0.9990740f, 0.0075000f);
        r75 = fmaf(r75, 0.9990750f, 0.0076000f);
        r76 = fmaf(r76, 0.9990760f, 0.0077000f);
        r77 = fmaf(r77, 0.9990770f, 0.0078000f);
        r78 = fmaf(r78, 0.9990780f, 0.0079000f);
        r79 = fmaf(r79, 0.9990790f, 0.0080000f);
        r80 = fmaf(r80, 0.9990800f, 0.0081000f);
        r81 = fmaf(r81, 0.9990810f, 0.0082000f);
        r82 = fmaf(r82, 0.9990820f, 0.0083000f);
        r83 = fmaf(r83, 0.9990830f, 0.0084000f);
        r84 = fmaf(r84, 0.9990840f, 0.0085000f);
        r85 = fmaf(r85, 0.9990850f, 0.0086000f);
        r86 = fmaf(r86, 0.9990860f, 0.0087000f);
        r87 = fmaf(r87, 0.9990870f, 0.0088000f);
        r88 = fmaf(r88, 0.9990880f, 0.0089000f);
        r89 = fmaf(r89, 0.9990890f, 0.0090000f);
        r90 = fmaf(r90, 0.9990900f, 0.0091000f);
        r91 = fmaf(r91, 0.9990910f, 0.0092000f);
        r92 = fmaf(r92, 0.9990920f, 0.0093000f);
        r93 = fmaf(r93, 0.9990930f, 0.0094000f);
        r94 = fmaf(r94, 0.9990940f, 0.0095000f);
        r95 = fmaf(r95, 0.9990950f, 0.0096000f);
    }

    // 使用所有变量，避免被编译器删除
    float sum = 0.0f;
    sum += r0;
    sum += r1;
    sum += r2;
    sum += r3;
    sum += r4;
    sum += r5;
    sum += r6;
    sum += r7;
    sum += r8;
    sum += r9;
    sum += r10;
    sum += r11;
    sum += r12;
    sum += r13;
    sum += r14;
    sum += r15;
    sum += r16;
    sum += r17;
    sum += r18;
    sum += r19;
    sum += r20;
    sum += r21;
    sum += r22;
    sum += r23;
    sum += r24;
    sum += r25;
    sum += r26;
    sum += r27;
    sum += r28;
    sum += r29;
    sum += r30;
    sum += r31;
    sum += r32;
    sum += r33;
    sum += r34;
    sum += r35;
    sum += r36;
    sum += r37;
    sum += r38;
    sum += r39;
    sum += r40;
    sum += r41;
    sum += r42;
    sum += r43;
    sum += r44;
    sum += r45;
    sum += r46;
    sum += r47;
    sum += r48;
    sum += r49;
    sum += r50;
    sum += r51;
    sum += r52;
    sum += r53;
    sum += r54;
    sum += r55;
    sum += r56;
    sum += r57;
    sum += r58;
    sum += r59;
    sum += r60;
    sum += r61;
    sum += r62;
    sum += r63;
    sum += r64;
    sum += r65;
    sum += r66;
    sum += r67;
    sum += r68;
    sum += r69;
    sum += r70;
    sum += r71;
    sum += r72;
    sum += r73;
    sum += r74;
    sum += r75;
    sum += r76;
    sum += r77;
    sum += r78;
    sum += r79;
    sum += r80;
    sum += r81;
    sum += r82;
    sum += r83;
    sum += r84;
    sum += r85;
    sum += r86;
    sum += r87;
    sum += r88;
    sum += r89;
    sum += r90;
    sum += r91;
    sum += r92;
    sum += r93;
    sum += r94;
    sum += r95;
    output[tid] = sum;
}

int main() {
    float *d_in, *d_out;
    CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i)
        h_in[i] = 1.0f + (i % 251) * 0.001f;

    CHECK(cudaMemcpy(d_in, h_in.data(),
                     N * sizeof(float),
                     cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 5; ++i)
        spill_kernel<<<N / BLOCK, BLOCK>>>(d_in, d_out, 100);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    for (int i = 0; i < 20; ++i)
        spill_kernel<<<N / BLOCK, BLOCK>>>(d_in, d_out, 100);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    float result;
    CHECK(cudaMemcpy(&result, d_out,
                     sizeof(float), cudaMemcpyDeviceToHost));

    printf("Average kernel time: %.4f ms\n", ms / 20);
    printf("First output: %.6f\n", result);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
}
