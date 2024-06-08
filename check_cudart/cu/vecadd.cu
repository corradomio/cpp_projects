extern "C" __global__ void VecAdd(float* A, float* B, float* C, float c)
{
    int i = blockDim.x*blockIdx.x + threadIdx.x ;
    C[i] = A[i] + B[i] + c;
}