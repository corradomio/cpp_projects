#ifndef CHECK_MATHEMATICA_LLU_LIBRARY_H
#define CHECK_MATHEMATICA_LLU_LIBRARY_H

#include "WolframLibrary.h"

extern "C" {
    int function(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res);
}

#endif //CHECK_MATHEMATICA_LLU_LIBRARY_H
