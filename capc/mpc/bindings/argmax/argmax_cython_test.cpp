#include "argmax_cython.hpp"
#include <stdlib.h>
#include <vector>
#include <iostream>

int main(int argc, char **argv) {
    int party = atoi(argv[1]);
    int port = 12345;
    std::vector<long long> array;

    array.push_back(3);
    array.push_back(4);
    array.push_back(1);

    std::cout << "Test argmax cpp code." << std::endl;
    long long result = argmax(party, port, array);
    std::cout << "result: " << result << std::endl;
}