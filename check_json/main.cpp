// rapidjson/example/simpledom/simpledom.cpp`
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <iostream>
#include <nlohmann/json.hpp>

using namespace rapidjson;
using namespace nlohmann;

int main() {
    //json j = json::parse("{ \"happy\": true, \"pi\": 3.141 }");

    json j = R"(
    {
        "k": 1,
        "m": {
            "a": 11,
            "b": 12
        }
    }
    )"_json;

    std::cout << j.dump() << std::endl;

    json m = j["m"];

    std::cout << m.dump() << std::endl;

}

int main1() {
    // 1. Parse a JSON string into DOM.
    const char* json = "{\"project\":\"rapidjson\",\"stars\":10}";
    Document d;
    d.Parse(json);

    // 2. Modify it by DOM.
    Value& s = d["stars"];
    s.SetInt(s.GetInt() + 1);

    // 3. Stringify the DOM
    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    d.Accept(writer);

    // Output {"project":"rapidjson","stars":11}
    std::cout << buffer.GetString() << std::endl;
    return 0;
}
