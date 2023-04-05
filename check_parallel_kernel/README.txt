cluster: distributed memory
    computer: multiple devices
              computer memory | multiple cores
        device: device memory | multiple cores
            grid





clusterDim.{x,y,z}
    gridIdx.{x,y,z}
    gridDim.{x,y,z}
        blockIdx.{x,y,z}
        blockDim.{x,y,z}
            threadIdx.{x,y,z}



__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}


__syncthreads()

__global__
__device__
__host__
__noinline__
__forceinline__
__shared__
__constant__
__grid_constant__
__restrict__


