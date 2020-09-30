
// demo-is-self-contained

#include "comb/comb-print.h"

#include "fxttypes.h"
#include "fxtio.h"
#include "jjassert.h"
#include "nextarg.h"



//% Eades-McKay (strong revolving door) order for combinations

//#define TIMING  // define to disable printing


ulong N;    // number of bits in words
ulong K;    // number of ones in words
ulong *rv;  // elements in combination at rv[1] ... rv[k]

ulong ct;   // count objects


void visit()
{
    cout << setw(4) << ct << ":";
    print_set("  ", rv+1, K, 1);
    print_set1_as_deltaset("  ", rv+1, K, N);
    cout << endl;
}
// -------------------------


void
comb_emk(ulong n, ulong k, bool z)
{
    if ( k==n )
    {
        for (ulong j=1; j<=k; ++j)  rv[j] = j;
        ++ct;
#ifndef TIMING
        visit();
#endif
        return;
    }

    if ( z )  // forward:
    {
        if ( (n>=2) && (k>=2) )  { rv[k] = n;  rv[k-1] = n-1;  comb_emk(n-2, k-2, z); }
        if ( (n>=2) && (k>=1) )  { rv[k] = n;  comb_emk(n-2, k-1, !z); }
        if ( (n>=1) )            { comb_emk(n-1, k, z); }
    }
    else     // backward:
    {
        if ( (n>=1) )            { comb_emk(n-1, k, z); }
        if ( (n>=2) && (k>=1) )  { rv[k] = n;  comb_emk(n-2, k-1, !z); }
        if ( (n>=2) && (k>=2) )  { rv[k] = n;  rv[k-1] = n-1;  comb_emk(n-2, k-2, z); }
    }
}
// -------------------------

void
comb_emk_compl(ulong n, ulong k, bool z)
{
    if ( (k==0) || (k==n) )
    {
        for (ulong j=1; j<=k; ++j)  rv[j] = j;
        ++ct;
        visit();
        return;
    }

    if ( z )  // forward:
    {
        if ( (n>=1) && (k>=1) )  { rv[k] = n;  comb_emk_compl(n-1, k-1, z); }     // 1
        if ( (n>=2) && (k>=1) )  { rv[k] = n-1;  comb_emk_compl(n-2, k-1, !z); }  // 01
        if ( (n>=2)  )           { comb_emk_compl(n-2, k-0, z); }                 // 00
    }
    else     // backward:
    {
        if ( (n>=2) )            { comb_emk_compl(n-2, k-0, z); }                 // 00
        if ( (n>=2) && (k>=1) )  { rv[k] = n-1;  comb_emk_compl(n-2, k-1, !z); }  // 01
        if ( (n>=1) && (k>=1) )  { rv[k] = n;  comb_emk_compl(n-1, k-1, z); }     // 1
    }
}
// -------------------------


int
main(int argc, char **argv)
{
    N = 7;
    NXARG(N, "Length of words");
    K = 3;
    NXARG(K, "Combinations (n choose k)");
    rv = new ulong[K+1];

    bool cq = 0;
    NXARG(cq, "Whether to generate complement");

    bool rq = 0;
    NXARG(rq, "Whether to reverse order");
    rq = !rq;

    jjassert( K <= N );

    ct = 0;

    if ( 0==cq )  comb_emk(N, K, rq);
    else          comb_emk_compl(N, K, rq);

    cout << "ct=" << ct << endl;

    delete [] rv;

    return 0;
}
// -------------------------

/*
Timing:  (Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz)

time ./bin 32 20
arg 1: 32 == N  [Length of words]  default=7
arg 2: 20 == K  [Combinations (n choose k)]  default=3
arg 3: 0 == cq  [Whether to generate complement]  default=0
arg 4: 0 == rq  [Whether to reverse order]  default=0
ct=225792840
./bin 32 20  1.08s user 0.00s system 99% cpu 1.080 total
 ==> 225792840/1.08 == 209,067,444 per second

time ./bin 32 12
arg 1: 32 == N  [Length of words]  default=7
arg 2: 12 == K  [Combinations (n choose k)]  default=3
arg 3: 0 == cq  [Whether to generate complement]  default=0
arg 4: 0 == rq  [Whether to reverse order]  default=0
ct=225792840
./bin 32 12  1.36s user 0.00s system 99% cpu 1.364 total
 ==> 225792840/1.36 == 166,024,147 per second
*/


/// Emacs:
/// Local Variables:
/// MyRelDir: "demo/comb"
/// makefile-dir: "../../"
/// make-target: "1demo DSRC=demo/comb/combination-emk-rec-demo.cc"
/// make-target2: "1demo DSRC=demo/comb/combination-emk-rec-demo.cc DEMOFLAGS=-DTIMING"
/// End:

