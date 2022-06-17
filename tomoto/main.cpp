#include <iostream>
#include <TopicModel/LDAModel.hpp>

using namespace tomoto;

int main() {
    std::cout << "Hello, World!" << std::endl;
    tomoto::TermWeight tw = TermWeight::one;
    tomoto::LDAArgs margs;
    tomoto::ITopicModel* inst = tomoto::ILDAModel::create((tomoto::TermWeight)tw, margs);
    return 0;
}
