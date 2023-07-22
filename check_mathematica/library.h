#ifndef CHECK_MATHEMATICA_LIBRARY_H
#define CHECK_MATHEMATICA_LIBRARY_H

// #include <stdint.h>
// #include <wstp.h>

#include "WolframLibrary.h"
#include "WolframIOLibraryFunctions.h"
#include "WolframNumericArrayLibrary.h"
#include "WolframCompileLibrary.h"

extern "C" {
    int demo1(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res);
    int demo2(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res);
    int demo3(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res);
}

#endif //CHECK_MATHEMATICA_LIBRARY_H
