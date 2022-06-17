//////////////////////////////////////////////////////////
// THIS FILE IS PART OF THE 'libagents' LIBRARY PACKAGE //
//////////////////////////////////////////////////////////

/*
 * The 'libagents' library is copyrighted software, all rights reserved.
 * (c) Information Technology Group - www.itgroup.ro
 * (c) Virgil Mager
 *
 * THE 'libagents' LIBRARY IS LICENSED FREE OF CHARGE, AND IT IS DISTRIBUTED "AS IS",
 * WITH EXPRESS DISCLAIMER OF SUITABILITY FOR ANY PARTICULAR PURPOSE,
 * AND WITH NO WARRANTIES OF ANY KIND, WHETHER EXPRESS OR IMPLIED,
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW.
 *
 * IN NO EVENT OTHER THAN WHEN MANDATED BY APPLICABLE LAW, SHALL THE AUTHOR(S),
 * THE PUBLISHER(S), AND/OR THE COPYRIGHT HOLDER(S) OF THE 'libagents' LIBRARY,
 * BE LIABLE FOR ANY CONSEQUENTIAL DAMAGES WHICH MAY DIRECTLY OR INDIRECTLY RESULT
 * FROM THE USE OR MISUSE, IN ANY WAY AND IN ANY FORM, OF THE 'libagents' LIBRARY.
 *
 * Permission is hereby granted by the 'libagents' library copyright holder(s)
 * to create and/or distribute any form of work based on, and/or dervied from,
 * and/or which uses in any way, the 'libagents' library, in full or in part, provided that
 * ANY SUCH DISTRUBUTION PROMINENTLY INCLUDES THE ORIGINAL 'libagents' LIBRARY PACKAGE.
 */

#ifndef __libagents_config_h__
#define __libagents_config_h__

/*************************************************************************
// g++/gnu/x64
// __id_t_defined MUST be #defined here, before the other system #includes
#define __id_t_defined
*************************************************************************/
#include <climits>
#include <sys/timeb.h>

#include <assert.h>
#include <cstdint>
#include <string>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

/********************************************************************************************************************
 !!! CRITICAL: in order to be able to compile libagents, the compiler #include path must contain the libagents folder
*********************************************************************************************************************/

#define libagents AGENTS_Lib

namespace AGENTS_Lib {

enum { // default values
    MESSAGE_BUFFER_SIZE=200,
    OS_TICK=1
};

enum { // bitiwise debugging flags
    SET=2,
    DEBUG_NONE=0,
    DEBUG_ASSERT=SET^0,
    DEBUG_MESSAGING=SET^1,
    DEBUG_ALL=-1
};

#ifndef force
#define force(boolexpr) AGENTS_Lib::break_on_false((boolexpr), __LINE__, __FILE__)
#endif
inline bool break_on_false(bool boolexpr, int line, std::string file) {
   if (!boolexpr) {
      printf("\nforce assertion failed at line %d in file %s\n\n", line, file.c_str());
      exit(1);
   }
   return true;
}

inline uint64_t time_ms() {
   struct timeb timeL;
   ftime(&timeL);
   return (uint64_t)timeL.time*1000+timeL.millitm;
}

inline int sleep_ms(int x) {
   return usleep(x*1000);
}

}

/******************************/

#endif
