//
// Created by Corrado Mio on 19/09/2020.
//

#ifndef CHECK_TRACKS_STRING_FOMAT_H
#define CHECK_TRACKS_STRING_FOMAT_H

#include <string>
#include <memory>

namespace stdx {

    template<typename ... Args>
    std::string format( const std::string& format, Args ... args )
    {
        size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
        if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
        std::unique_ptr<char[]> buf( new char[ size ] );
        snprintf( buf.get(), size, format.c_str(), args ... );
        return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
    }


}

#endif //CHECK_TRACKS_STRING_FOMAT_H
