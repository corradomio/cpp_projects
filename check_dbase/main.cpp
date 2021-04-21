#include <iostream>
//#include <libpq-fe.h>
#include <cstdio>
#include <pqxx/pqxx>
#include <mysql.h>

//int main()
//{
//    printf("MySQL client version: %s\n", mysql_get_client_info());
//    MYSQL *con = mysql_init(NULL);
//
//    if (con == NULL)
//    {
//        fprintf(stderr, "%s\n", mysql_error(con));
//        exit(1);
//    }
//
//    if (mysql_real_connect(con, "localhost", "root", "",
//                           "test", 0, NULL, 0) == NULL)
//    {
//        fprintf(stderr, "%s\n", mysql_error(con));
//        mysql_close(con);
//        exit(1);
//    }
//
//    if (mysql_query(con, "CREATE DATABASE testdb"))
//    {
//        fprintf(stderr, "%s\n", mysql_error(con));
//        mysql_close(con);
//        exit(1);
//    }
//
//    mysql_close(con);
//    exit(0);
//
//    exit(0);
//}

int main()
{
    try
    {
        pqxx::connection C("hostaddr=192.168.161.129 user=osmuser password=osmuser dbname=osm");
        std::cout << "Connected to " << C.dbname() << std::endl;
        pqxx::work W{C};

        pqxx::result R{W.exec("select * from abu_dhabi_agents_state limit 20")};

        std::cout << "Found " << R.size() << "employees:\n";
        for (auto row: R)
            std::cout << row[1].c_str() << '\n';

        std::cout << "OK.\n";
    }
    catch (std::exception const &e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}

//void do_exit(PGconn *conn) {
//
//    PQfinish(conn);
//    exit(1);
//}
//
//
//int main() {
//    int lib_ver = PQlibVersion();
//
//    printf("Version of libpq: %d\n", lib_ver);
//
//    PGconn *conn = PQconnectdb("hostaddr=192.168.161.129 user=osmuser password=osmuser dbname=osm");
//    if (PQstatus(conn) == CONNECTION_BAD) {
//
//        fprintf(stderr, "Connection to database failed: %s\n",
//                PQerrorMessage(conn));
//        do_exit(conn);
//    }
//
//    int ver = PQserverVersion(conn);
//
//    printf("Server version: %d\n", ver);
//
//    PQfinish(conn);
//
//
//    return 0;
//}
