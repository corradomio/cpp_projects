
#include <served/served.hpp>

/* hello_world example
 *
 * This is the most basic example of served in action.
 */
int main4(int, char const**)
{
    served::multiplexer mux;

    mux.handle("/hello")
        .get([](served::response & res, const served::request &) {
            res << "Hello world";
        });

    std::cout << "Try this example with:" << std::endl;
    std::cout << " curl http://localhost:8123/hello" << std::endl;

    served::net::server server("127.0.0.1", "8123", mux);
    server.run(10); // Run with a pool of 10 threads.

    return 0;
}
