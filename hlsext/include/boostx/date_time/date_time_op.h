//
// Created by Corrado Mio on 18/09/2020.
//

#ifndef HLSEXT_DATE_TIME_OP_H
#define HLSEXT_DATE_TIME_OP_H

#include <boost/date_time.hpp>

using namespace boost::posix_time;

namespace std {

    inline string to_string(time_duration td) {
        return to_simple_string(td);
    }

    inline string to_string(ptime t) {
        return to_simple_string(t);
    }

}

namespace boost {
namespace posix_time {

    ptime to_ptime(std::string s) {
        return time_from_string(s);
    }

    time_duration to_duration(std::string s) {
        return duration_from_string(s);
    }

    /**
     * Multiply a time_duration by a factor
     * @param f factor
     * @param td time duration
     * @return a time duration
     */
    inline time_duration operator*(const double f, const time_duration& td) {
        return {0,0, (long)(f*td.total_seconds())};
    }

    /**
     * Ration between two time durations
     * @param num numerator
     * @param den denominator
     * @return ratio between two time durations
     */
    inline double operator/(const time_duration& num, const time_duration& den) {
        return (num.total_seconds()+0.)/(den.total_seconds()+0.);
    }

} }

#endif //HLSEXT_DATE_TIME_OP_H
