#include <iostream>
#include <cmath>
#include <ctime>
#include "tbb/task_scheduler_init.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/partitioner.h"

using namespace tbb;

static const size_t N=200000000;

float timestamp() {
    return float(clock())/CLOCKS_PER_SEC;
}

float* Initialize(){
    float *a = new float[N];
    for (size_t i=0; i<N; ++i)
        a[i] = float(i)+1.0;

    return a;
}


float Foo(float x){
    return std::sin(x);
}

void SerialApplyFoo( float a[], size_t n ) {
    for( size_t i=0; i<n; ++i )
        Foo(a[i]);
}

class ApplyFoo {
    float *const my_a;
public:
    void operator( )( const blocked_range<size_t>& r ) const {
        float *a = my_a;
        for( size_t i=r.begin(); i!=r.end( ); ++i )
            Foo(a[i]);
    }
    ApplyFoo( float a[] ) :
            my_a(a)
    {}
};

void ParallelApplyFoo( float a[], size_t n ) {
    parallel_for(blocked_range<size_t>(0,n), ApplyFoo(a), auto_partitioner( ) );
}

struct Average {
    float* input;
    float* output;
    void operator( )( const blocked_range<int>& range ) const {
        for( int i=range.begin(); i!=range.end( ); ++i )
            output[i] = (input[i-1]+input[i]+input[i+1])*(1/3.0f);
    }
};
// Note: The input must be padded such that input[-1] and input[n]
// can be used to calculate the first and last output values.
void ParallelAverage( float* output, float* input, size_t n ) {
    Average avg;
    avg.input = input;
    avg.output = output;
    parallel_for( blocked_range<int>( 0, n, 1000 ), avg );
}

int main( ) {
    task_scheduler_init init;
    float start = timestamp();

    std::cout << "start" << std::endl;

    float *a = Initialize();

    //SerialApplyFoo(a, N);
    ParallelApplyFoo(a, N);

    std::cout << "done " << (timestamp() - start) << "s" << std::endl;
    return 0;
}