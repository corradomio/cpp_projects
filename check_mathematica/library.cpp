#include "library.h"
#include <LLU/LLU.h>
#include <string>
#include <numeric>

DLLEXPORT mint WolframLibrary_getVersion( ) {
    return WolframLibraryVersion;
}

DLLEXPORT int WolframLibrary_initialize( WolframLibraryData libData) {
    return 0;
}

DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData) {
    return;
}


DLLEXPORT int demo1(WolframLibraryData libData,
                       mint Argc,
                       MArgument *Args,
                       MArgument Res) {
    mint I0;
    mint I1;
    I0 = MArgument_getInteger(Args[0]);
    I1 = I0 + 1;
    MArgument_setInteger(Res, I1);
    return LIBRARY_NO_ERROR;
}

#define MTensor_new (libData->MTensor_new)

DLLEXPORT int demo2(WolframLibraryData libData,
                       mint Argc,
                       MArgument *Args,
                       MArgument Res) {
    mint I0 = MArgument_getInteger(Args[0]);
    mint I1 = I0*I0;
//    MArgument_setInteger(Res, I1);

    MTensor T0;
    mint type = MType_Real;
    mint dims[2];
    mint rank = 2;
    dims[0] = 5;
    dims[1] = 5;
    int err = MTensor_new( type, rank, dims, &T0);

    double* p = (double*)((*T0).data);
    for (int i=0; i<25; ++i)
        p[i] = i;

    MArgument_setMTensor(Res, T0);
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT int demo3(WolframLibraryData libData, mint Argc, MArgument *Args, MArgument Res) {
    auto err = LLU::ErrorCode::NoError;
    try {
        // Create manager object
        LLU::MArgumentManager mngr {libData, Argc, Args, Res};

        // Read string and NumericArray arguments
        auto string = mngr.getString(0);
        auto counts = mngr.getNumericArray<std::uint8_t>(1);

        // check NumericArray rank
        if (counts.rank() != 1) {
            LLU::ErrorManager::throwException(LLU::ErrorName::RankError);
        }

        // check if NumericArray length is equal to input string length
        if (counts.size() != string.size()) {
            LLU::ErrorManager::throwException(LLU::ErrorName::DimensionsError);
        }

        // before we allocate memory for the output string, we have to sum all NumericArray elements
        // to see how many bytes are needed
        auto sum = std::accumulate(std::cbegin(counts), std::cend(counts), static_cast<size_t>(0));

        // allocate memory for the output string
        std::string outString;
        outString.reserve(sum);

        // populate the output string
        for (mint i = 0; i < counts.size(); i++) {
            outString.append(std::string(counts[i], string[i]));
        }

        // clean up and set the result
        mngr.set(std::move(outString));
    } catch (const LLU::LibraryLinkError& e) {
        err = e.which();
    }
    return err;
}