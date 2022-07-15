#include "argmax_cython.hpp"
#include "emp-sh2pc/emp-sh2pc.h"
#include <iostream>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>

using namespace std;
using namespace emp;

const int BITLENGTH = 64;
const int MOD_LENGTH = 38;//should be *less* than BITLENGTH
// TODO: tune the mod_length and add tests for the boundaries in the python code

Integer mod(const Integer &a) {
    return a & Integer(BITLENGTH, (1 << MOD_LENGTH) - 1, PUBLIC);
}

long long argmax(vector<long long> &data, vector<long long> &rr, int party) {
    // argmax((alice+bob) mod 2^MOD_LENGTH )
    vector <Integer> alice;
    vector <Integer> bob;
    if (party == ALICE) {
        // cout << "In ALICE." << endl;
        for (auto v : data) {
            alice.push_back(Integer(BITLENGTH, v + 8388608, ALICE));
            // cout << "data item: " << v << endl;
            rr.push_back(-v);
        }
        for (int i = 0; i < data.size(); ++i)
            bob.push_back(Integer(BITLENGTH, 0, BOB));
    } else {
        // cout << "In BOB." << endl;
        for (int i = 0; i < data.size(); ++i)
            alice.push_back(Integer(BITLENGTH, 0, ALICE));
        for (auto v : data)
            bob.push_back(Integer(BITLENGTH, v + 8388608, BOB));
    }

    Integer index(BITLENGTH, 0, PUBLIC);
    Integer max_value = mod(alice[0] + bob[0]);
    // cout << "value: " << max_value.reveal<uint64_t>(PUBLIC) << endl;
    for (int i = 1; i < data.size(); ++i) {
        Integer value = mod(alice[i] + bob[i]);
        // cout << "value: " << value.reveal<uint64_t>(PUBLIC) << endl;
        Bit greater = value > max_value;
        index = index.select(greater, Integer(BITLENGTH, i, PUBLIC));
        max_value = max_value.select(greater, value);
    }
    long long res = index.reveal<uint64_t>(PUBLIC);
    return res;
}

long long argmax(int party, int port, vector<long long> &array) {
    cout << "In argmax library." << endl;
    int res;

    if (party == ALICE) {
        vector<long long> noise = array;
        vector<long long> rr;
        res = argmax(noise, rr, party);
    } else {
        vector<long long> logits = array;
        vector<long long> temp;
        res = argmax(logits, temp, party);
    }

    return res;
}

void read_vector(int argc, char **argv, vector<long long> &array) {
    for (int index = 4; index < argc; ++index) {
        array.push_back(atoi(argv[index]));
    }
}

int main(int argc, char **argv) {
    // USAGE: party number, port, ip_address, input_array
    // if party == 1 (ALICE) pass noise values.
    // if party == 2 (BOB) pass logits.
    int party;
    int port;
    // https://github.com/emp-toolkit/emp-tool/blob/b07a7d9ab3053a3e16991751402742d418377f63/emp-tool/utils/utils.h
    parse_party_and_port(argv, &party, &port);
    const char *ip_address = (party == ALICE ? nullptr : argv[3]);
    NetIO *io = new NetIO(party == ALICE ? nullptr : ip_address, port);

    vector<long long> array;
    read_vector(argc, argv, array);

    auto prot = setup_semi_honest(io, party);
    prot->set_batch_size(1024 * 1024);//set it to number of bits in BOB's input

    long long res = argmax(party, port, array);
    cout << "party: " << party << " res: " << res << endl;
}
