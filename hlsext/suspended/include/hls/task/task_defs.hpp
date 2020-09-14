//
// Created by Corrado Mio on 28/02/2016.
//

#ifndef HLS_TASK_DEFS_HPP
#define TBBTEST_STDDEF_HPP

namespace hls {
namespace task {
namespace internal {

    class no_copy {
        no_copy(const no_copy&);
    public:
        no_copy(){}
    };

    class no_assign {
        no_assign& operator =(const no_assign&);
    public:
        no_assign(){}
    };

}


    class split { };


}};

#endif // HLS_TASK_DEFS_HPP
