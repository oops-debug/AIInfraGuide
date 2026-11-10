#include <cstdio>
__global__ void helloKernel(){
    printf("hello from GPU thread %d in block %d!\n",
            threadIdx.x, blockIdx.x);
}

int main(){
    printf("Launching kernel...\n");
    helloKernel<<<2,4>>>();
    cudaDeviceSynchronize();
    printf("done!/n");
    return 0;
}