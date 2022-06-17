#include <stdio.h>
#include <igraph.h>
#include "igraph.hpp"


namespace ig = igraph;


int main() {

    //ig::graph_t g1(10);
    //ig::graph_t g2(g1);
    //ig::graph_t g3;
    //
    //g3 = g2;

    int v[5]={8};
    for (int i=0; i<5; ++i)
        printf("%d\n", v[i]);

    return 0;
}

int main1(void) {
    igraph_real_t diameter;
    igraph_t graph;
    igraph_rng_seed(igraph_rng_default(), 42);
    igraph_erdos_renyi_game(&graph, IGRAPH_ERDOS_RENYI_GNP, 1000, 5.0 / 1000,
                            IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);
    igraph_diameter(&graph, &diameter, 0, 0, 0, IGRAPH_UNDIRECTED, 1);
    printf("Diameter of a random graph with average degree 5: %d\n",
           (int) diameter);
    igraph_destroy(&graph);
    return 0;
}