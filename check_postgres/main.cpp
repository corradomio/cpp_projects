#include <iostream>
#include <string>
#include <unordered_map>
#include <pqxx/pqxx>
#include "boost/date_time/gregorian/gregorian.hpp"
#include "boost/date_time/posix_time/posix_time.hpp"

using namespace boost::gregorian;
using namespace boost::posix_time;

void test1(pqxx::connection& c) {
    //pqxx::connection c{"postgresql://osmuser:osmuser@192.168.161.129/osm"};
    pqxx::work t{c};

    // Normally we'd query the DB using txn.exec().  But for querying just one
    // single value, we can use txn.query_value() as a shorthand.
    //
    // Use txn.quote() to escape and quote a C++ string for use as an SQL string
    // in a query's text.
    int count = t.query_value<int>(
        "SELECT count(uid) "
        "FROM abu_dhabi_tracks "
        "WHERE uid =1");

    std::cout << "Count # " << count << '\n';

    // Make our change definite.
    t.commit();
}

struct coords_t {
    double lon;
    double lat;
};


bool operator ==(const coords_t& c1, const coords_t& c2) {
    return c1.lon == c2.lon && c1.lat == c2.lat;
}


template<>
struct std::hash<coords_t>
{
    size_t
    operator()(coords_t const& c) const noexcept
    {
        std::size_t h1 = std::hash<double>{}(c.lon);
        std::size_t h2 = std::hash<double>{}(c.lat);
        return h1 ^ (h2 << 1);
    }
};


void test2(pqxx::connection& c) {
    pqxx::work t{c};

    ptime ts(date(2019,1,1), time_duration(0,30,0));
    std::string p1 = to_iso_extended_string(ts);

    std::cout << "ts: " << p1 << std::endl;

    pqxx::result rs = t.exec_prepared("sel", t.quote(p1));

    std::unordered_map<coords_t, std::vector<int>> users;

    long count = 0;
    for(auto it : rs) {
        int uid;
        coords_t c;
        double lon;
        double lat;

        it[0].to(uid);
        it[1].to(c.lon);
        it[1].to(c.lat);

        users[c].push_back(uid);

        //std::cout << uid << ":" << lon << "," << lat << std::endl;
        ++count;
    }
    t.commit();

    std::cout << "done " << count << std::endl;
    std::cout << "  " << users.size() << std::endl;

    count = 0;
    for (auto it : users) {
        count += it.second.size();
    }
    std::cout << "  " << count << std::endl;


}


int main(int, char *argv[])
{
    std::string conn = "postgresql://osmuser:osmuser@localhost/osm";
    pqxx::connection c{conn};
    c.prepare("sel", "SELECT uid,ilon,ilat FROM abu_dhabi_tracks "
                     "WHERE tid BETWEEN 0 AND 288");

    test1(c);
    test2(c);
}