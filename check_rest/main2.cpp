#include <iostream>
#include "restc-cpp/restc-cpp.h"

using namespace std;
using namespace restc_cpp;

void DoSomethingInteresting2(Context& ctx) {
    // Here we are in a co-routine, running in a worker-thread.

    // Asynchronously connect to a server and fetch some data.
    auto reply = ctx.Get("http://jsonplaceholder.typicode.com/posts/1");

    // Asynchronously fetch the entire data-set and return it as a string.
    auto json = reply->GetBodyAsString();

    // Just dump the data.
    cout << "Received data: " << json << endl;
}

int main2() {
    auto rest_client = RestClient::Create();

    // Call DoSomethingInteresting as a co-routine in a worker-thread.
    rest_client->Process(DoSomethingInteresting2);

    // Wait for the coroutine to finish, then close the client.
    rest_client->CloseWhenReady(true);

    return 0;
}