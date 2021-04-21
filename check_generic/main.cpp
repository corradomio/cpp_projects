#include<stdio.h>
#include<stdlib.h>

const int N = 1000;
int V[N];

int rmin(const int V[], int i) {
    if (i < 0)
        return __INT_MAX__;
    else if (i == 0)
        return V[0];
    int fmin = rmin(V, i-1);
    if (fmin < V[i])
        return fmin;
    else
        return V[i];
}


int main(int argc, char** argv) {

    // Inizializzazione del generatore di numeri casuali
    // 'argv' ha un indirizzo sempre diverso
    // idea: usare l'indirizzo come un long da usare con srand()

    union {
        void* ptr;
        long ival;
    } init;
    init.ptr = argv;

    printf("init srand: %ld\n", init.ival);
    srand(init.ival);

    // inizializzazione di min con il PIU' ALTO valore per un intero
    int min = __INT_MAX__;

    // inizializzazione del vettore e identificazione del valore minimo
    for(int i=0;i<N; ++i) {
        V[i] = 1 + rand() % 1000;
        if (V[i] < min)
            min = V[i];
    }

    printf("  real min: %d\n", min);

    // 'rmin' minimo cercato in modo ricorsivo

    int min_found = rmin(V, N-1);

    // minimo trovato
    printf(" min found: %d\n", min_found);

   return 0;
}