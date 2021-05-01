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
        std::string str = to_iso_extended_string(t);
        return str.replace(str.find("T"), 1, " ");
    }

    inline string to_string(ptime::date_type d) {
        std::string str = to_simple_string(d);
        return str;
    }

}

namespace boost {
namespace posix_time {

    inline ptime to_ptime(std::string s) {
        return time_from_string(s);
    }

    inline time_duration to_duration(std::string s) {
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


    inline long days_diff(const ptime& dt0, const ptime& dte) {
        return (dte - dt0).hours()/24;
    }

    inline long days_diff(const date_range_t& date_range) {
        return days_diff(date_range.first, date_range.second);
    }

    inline ptime days_add(const ptime& dt0, int days) {
        return dt0 + time_duration(24*days, 0, 0);
    }

} }

#endif //HLSEXT_DATE_TIME_OP_H
