"""Cython interface definition for MPC."""
from libcpp.vector cimport vector

cdef extern from "argmax_cython.hpp":
    long long argmax(int party, int port, vector[long long] array)

def pyargmax(party: int, port: int, values: list):
    return argmax(party, port, values)

