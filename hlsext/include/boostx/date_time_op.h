//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_DATE_TIME_OP_H
#define HLSEXT_DATE_TIME_OP_H

#include <boost/date_time.hpp>

using namespace boost::posix_time;

namespace boost {
namespace posix_time {

    inline time_duration operator*(const double f, const time_duration& td) {
        return time_duration(0,0, f*td.total_seconds());
    }

    inline double operator/(const time_duration& num, const time_duration& den) {
        return (num.total_seconds()+0.)/(den.total_seconds()+0.);
    }

} }

#endif //HLSEXT_DATE_TIME_OP_H
