#include <cstdio>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n",
           prop.major, prop.minor);

    printf("Max threads per block: %d\n",
           prop.maxThreadsPerBlock);

    printf("Max threads per SM: %d\n",
           prop.maxThreadsPerMultiProcessor);

    printf("Max block dimensions: %d, %d, %d\n",
           prop.maxThreadsDim[0],
           prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);

    printf("Max grid dimensions: %d, %d, %d\n",
           prop.maxGridSize[0],
           prop.maxGridSize[1],
           prop.maxGridSize[2]);
       printf("SM count: %d\n",
       prop.multiProcessorCount);
       printf("Warp size: %d\n", prop.warpSize);
    return 0;
}