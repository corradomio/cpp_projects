//
// Created by Corrado Mio on 06/04/2020.
//

#ifndef CHECK_CUDA_CUGRAPH_H
#define CHECK_CUDA_CUGRAPH_H

struct vertex_t {
    int degree;
    int color;
    int bck_color;
    int ref_edges;
};

struct edge_t {
    int ref_vertex;
    float weight;
};

struct cugraph_t {
    vertex_t* vertices;
    size_t n_vertices;

    edge_t* edges;
    size_t n_edges;
};


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#endif //CHECK_CUDA_CUGRAPH_H
