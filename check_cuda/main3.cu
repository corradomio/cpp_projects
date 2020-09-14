#include <iostream>
#include <stdio.h>
#include <curand.h>
#include "cugraph.h"
#include <igraph/igraph.h>

__device__ static cugraph_t dg;
__device__ static curandGenerator_t rnd;

//

__device__ int randint() {

}

__device__ void graph_init(vertex_t* vertices, size_t n_vertices, edge_t* edges, size_t n_edges) {
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid != 0)
        return;
    dg.vertices = vertices;
    dg.n_vertices = n_vertices;
    dg.edges = edges;
    dg.n_edges = n_edges;

    //curandCreateGenerator(&rnd, (curandRngType_t)CURAND_RNG_PSEUDO_MT19937);
}

__global__ void cugraph_init_coloring() {
    size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    if (tid >= dg.n_vertices)
        return;

    dg.vertices[tid].degree = tid%5;
    dg.vertices[tid].color = tid%7;
}

__global__ void cugraph_processing(vertex_t* vertices, size_t n_vertices, edge_t* edges, size_t n_edges)
{
    //size_t tid = threadIdx.x + blockIdx.x*blockDim.x;
    //if (tid >= n_vertices)
    //    return;

    graph_init(vertices, n_vertices, edges, n_edges);
    __syncthreads();

    //cugraph_init_coloring();
    //__syncthreads();
}

//void* cudaMallocAndUpload(void* host, size_t size) {
//    void* device;
//    HANDLE_ERROR(cudaMalloc(&device, size));
//    HANDLE_ERROR(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
//    //cudaMallocManaged(&device, size);
//    return device;
//}

//cudaError_t cudaDownloadAndFree(void* host, void* device, size_t size) {
//    HANDLE_ERROR(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
//    return cudaFree(device);
//}

void cugraph_allocate(cugraph_t& g, size_t n_vertices, size_t n_edges) {
    g.n_vertices = n_vertices;
    g.n_edges = n_edges;
    HANDLE_ERROR(cudaMallocManaged(&g.vertices, n_vertices*sizeof(vertex_t)));
    HANDLE_ERROR(cudaMallocManaged(&g.edges, n_edges*sizeof(edge_t)));
}

void graph_free(cugraph_t g) {
    HANDLE_ERROR(cudaFree(g.vertices));
    HANDLE_ERROR(cudaFree(g.edges));
}

void dump_colors(cugraph_t& g) {
    printf("Colors\n");
    for(int i=0; i<10; ++i)
        printf("  v[%d]={d:%d,c:%d}\n", i,g.vertices[i].degree, g.vertices[i].color);
}

void cugraph_convert(cugraph_t cug, igraph_t ig) {

}


int	main(void)
{
    int n_vertices = 1000;
    int n_edges = 10000;
    
    cugraph_t g;
    cugraph_allocate(g, n_vertices, n_edges);

    dump_colors(g);

    int n_threads = 512;
    int n_blocks = (n_vertices + n_threads - 1)/n_threads;
    cugraph_processing<<<n_blocks, n_threads>>>(g.vertices, g.n_vertices, g.edges, g.n_edges);
    HANDLE_ERROR(cudaDeviceSynchronize());

    cugraph_init_coloring<<<n_blocks, n_threads>>>();
    HANDLE_ERROR(cudaDeviceSynchronize());

    dump_colors(g);

    graph_free(g);

    printf("All threads are finished!\n");
    return	0;
}
