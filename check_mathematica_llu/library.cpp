#include "library.h"
#include <LLU/LLU.h>

EXTERN_C DLLEXPORT int function(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
    LLU::MArgumentManager mngr {libData, Argc, Args, Res};
    auto n1 = mngr.get<mint>(0);  // get first (index = 0) argument, which is of type mint
    auto n2 = mngr.get<mint>(1);  // get second argument which is also an integer

    mngr.set(n1 + n2);  // set the sum of arguments to be the result
    return LLU::ErrorCode::NoError;
    return 0;
}
