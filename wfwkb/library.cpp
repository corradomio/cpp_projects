#include "library.h"
#include <WolframLibrary.h>
#include <WolframCompileLibrary.h>
#include <LLU/LLU.h>
#include <LLU/LibraryLinkFunctionMacro.h>
#include "LLULogger.h"
// #include <LLU/ErrorLog/Logger.h>

LLU_LOG_INITIALIZE("wkb", "wkblog.log")


DLLEXPORT mint WolframLibrary_getVersion( ) {
    return WolframLibraryVersion;
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
    LLU::LibraryData::setLibraryData(libData);

    LLU_LOG_LEVEL(LLU_LEVEL_DEBUG);
    LLU_DEBUG("WolframLibrary_initialize");

    return LLU::ErrorCode::NoError;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize( WolframLibraryData libData) {
    LLU_DEBUG("WolframLibrary_uninitialize");
    return;
}


///
/// wkb_parser(ba_ByteArray) -> {Line[{{x1,y1}, ...}], ...}.
///
LLU_LIBRARY_FUNCTION(wkb_parser) {
    LLU_DEBUG("wkb_parser");

    /*
        struct st_MNumericArray
        {
            mreal prec;
            mint* dims;
            mint rank;
            type_t tensor_property_type;
            uint32_t flags;
            type_t data_type;
            mint nelems;
            void* data;
            umint refcount;
        };
    */

    MNumericArray na = mngr.getMNumericArray(0);
    mngr.setMNumericArray(na);

    LLU_DEBUG("prec:{:f}, rank:{:d}, tptype:{:d}, flags:{:d}, dtype:{:d}, nelem:{:d}, rc:{:d}",
              na->prec,
              na->rank,
              na->tensor_property_type,
              (int)na->flags,       //uint
              na->data_type,
              na->nelems,
              (int)na->refcount);   //uint

}

// none     0
// int8
// uint8
// int16
// uint16
// int32
// uint32
// int64
// uint64
// float
// double

enum ParseType {
    None=0,
    Integer8,       // 1
    Integer16,      // 2
    Integer32,      // 3
    Integer64,      // 4
    Unsigned8,      // 5
    Unsigned16,     // 6
    Unsigned32,     // 7
    Unsigned64,     // 8
    Real32,         // 9
    Real64          // 10
};

LLU_LIBRARY_FUNCTION(bin_parse) {
    // ByteArray, pos, type
    MNumericArray na = mngr.getMNumericArray(0);
    mint pos  = mngr.get<mint>(1);
    mint type = mngr.get<mint>(2);
    union {
         int8_t  i8;
         int16_t i16;
         int32_t i32;
         int64_t i64;
        uint8_t  u8;
        uint16_t u16;
        uint32_t u32;
        uint64_t u64;
        float    f;
        double   d;
        char c[8]; } value;

    switch(type) {
        case None:
            break;
        case Integer8:
            value.c[0] = ((mint*)(na->data))[pos++];
            mngr.setInteger(value.i8);
            break;
        case Unsigned8:
            value.c[0] = ((mint*)(na->data))[pos++];
            mngr.setInteger(value.u8);
            break;
        case Integer16:
            value.c[0] = ((mint*)(na->data))[pos++];
            value.c[1] = ((mint*)(na->data))[pos++];
            mngr.setInteger(value.i16);
            break;
        case Unsigned16:
            value.c[0] = ((mint*)(na->data))[pos++];
            value.c[1] = ((mint*)(na->data))[pos++];
            mngr.setInteger(value.u16);
            break;
        case Integer32:
            for(int i=0; i<4; ++i)
                value.c[i] = ((mint*)(na->data))[pos++];
            mngr.setInteger(value.i32);
            break;
        case Unsigned32:
            for(int i=0; i<4; ++i)
                value.c[i] = ((mint*)(na->data))[pos++];
            mngr.setInteger(value.u32);
            break;
        case Integer64:
            for(int i=0; i<8; ++i)
                value.c[i] = ((mint*)(na->data))[pos++];
            mngr.setInteger(value.i64);
            break;
        case Unsigned64:
            for(int i=0; i<8; ++i)
                value.c[i] = ((mint*)(na->data))[pos++];
            mngr.setInteger(value.u64);
            break;
        case Real32:
            for(int i=0; i<4; ++i)
                value.c[i] = ((mint*)(na->data))[pos++];
            mngr.setReal(value.f);
            break;
        case Real64:
            for(int i=0; i<8; ++i)
                value.c[i] = ((mint*)(na->data))[pos++];
            mngr.setReal(value.d);
            break;
        default:
            mngr.set("Error");
    }
}

// void impl_wkb_parser(LLU::MArgumentManager&);
// LIBRARY_LINK_FUNCTION(wkb_parser) {
//     // auto err = LLU::ErrorCode::NoError;
//     // try {
//     //     logger->info("wkb_parser: 1");
//     //     LLU::MArgumentManager mngr {libData, Argc, Args, Res};
//     //     logger->info("wkb_parser: 2");
//     //     impl_wkb_parser(mngr);
//     //     logger->info("wkb_parser: 3");
//     // } catch (const LLU::LibraryLinkError& e) {
//     //     logger->error("wkb_parser: 4");
//     //     err = e.which();
//     // } catch (...) {
//     //     logger->error("wkb_parser: 5");
//     //     err = LLU::ErrorCode::FunctionError;
//     // }
//     // logger->info("wkb_parser: 6");
//     // return err;
//
//     logger->info("wkb_parser: 1");
//     return LLU::ErrorCode::NoError;
// }
//
// void impl_wkb_parser(LLU::MArgumentManager& mngr) {
//     logger->info("wkb_parser: 7");
//     MNumericArray na = mngr.getMNumericArray(0);
//     mngr.setMNumericArray(na);
//     logger->info("wkb_parser: 8");
// }

// EXTERN_C DLLEXPORT int LogDemo(WolframLibraryData libData, mint argc, MArgument* args, MArgument res) {
//     LLU_DEBUG("Library function entered with ", argc, " arguments.");
//     auto err = LLU::ErrorCode::NoError;
//     try {
//         LLU::MArgumentManager mngr(libData, argc, args, res);
//         auto index = mngr.getInteger<mint>(0);
//         if (index >= argc) {
//             LLU_WARNING("Index ", index, " is too big for the number of arguments: ", argc, ". Changing to ", argc - 1);
//             index = argc - 1;
//         }
//         auto value = mngr.getInteger<mint>(index);
//         mngr.setInteger(value);
//     } catch (const LLU::LibraryLinkError& e) {
//         LLU_ERROR("Caught LLU exception ", e.what(), ": ", e.debug());
//         err = e.which();
//     }
//     return err;
// }
