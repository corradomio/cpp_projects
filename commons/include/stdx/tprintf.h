//
// Created by Corrado Mio on 20/03/2024.
//

#ifndef STDX_TPRINTF_H
#define STDX_TPRINTF_H

namespace stdx {

    bool can_tprint(bool force=false);
    void tprintf(const char *__format, ...);
}

#endif //STDX_TPRINTF_H
