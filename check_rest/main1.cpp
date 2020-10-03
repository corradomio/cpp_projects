#include <iostream>
#define BOOST_LOG_DYN_LINK 1
#include <boost/lexical_cast.hpp>
#include <boost/fusion/adapted.hpp>

#include <restc-cpp/restc-cpp.h>
#include <restc-cpp/SerializeJson.h>
#include <restc-cpp/RequestBuilder.h>
#include <restc-cpp/IteratorFromJsonSerializer.h>

using namespace std;
using namespace restc_cpp;


// For entries received from http://jsonplaceholder.typicode.com/posts
struct Post {
    int userId = 0;
    int id = 0;
    string title;
    string body;
};

BOOST_FUSION_ADAPT_STRUCT(
    Post,
    (int, userId)
        (int, id)
        (string, title)
        (string, body)
)

void DoSomethingInteresting(Context& ctx) {

    try {
        // Asynchronously fetch the entire data-set, and convert it from json
        // to C++ objects was we go.
        // We expcet a list of Post objects
        list<Post> posts_list;
        SerializeFromJson(posts_list,
                          ctx.Get("http://jsonplaceholder.typicode.com/posts"));

        // Just dump the data.
        for(const auto& post : posts_list) {
            clog << "Post id=" << post.id << ", title: " << post.title << endl;
        }

    } catch (const exception& ex) {
        clog << "Caught exception: " << ex.what() << endl;
    }
}

int main1() {
    // Fetch a list of records asyncrouesly, one by one.
    // This allows us to process single items in a list
    // and fetching more data as we move forward.
    // This works basically as a database cursor, or
    // (literally) as a properly implemented C++ input iterator.

    // Create the REST clent
    auto rest_client = RestClient::Create();

    // Run our example in a lambda co-routine
    rest_client->Process([&](Context& ctx) {
        // This is the co-routine, running in a worker-thread

        // Construct a request to the server
        auto reply = RequestBuilder(ctx)
            .Get("http://jsonplaceholder.typicode.com/posts/")

                // Add some headers for good taste
            .Header("X-Client", "RESTC_CPP")
            .Header("X-Client-Purpose", "Testing")

                // Send the request
            .Execute();

        // Instatiate a serializer with begin() and end() methods that
        // allows us to work with the reply-data trough a C++
        // input iterator.
        IteratorFromJsonSerializer<Post> data{*reply};

        // Iterate over the data, fetch data asyncrounesly as we go.
        for(const auto& post : data) {
            cout << "Item #" << post.id << " Title: " << post.title << endl;
        }
    });


    // Wait for the request to finish
    rest_client->CloseWhenReady(true);

    return 0;
}
