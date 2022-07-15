"""
Example cython interface definition.
Definition of the Python bindings.
"""

# Tell Cython that you’re using cppmult() from cppmult.hpp.
cdef extern from "cppmult.hpp":
    # The function declarations below are also found in the cppmult.hpp file.
    # This ensures that your Python bindings are built against the same
    # declarations as the C++ code.
    float cppmult(int int_param, float float_param)

# Create a wrapper standard Python function, pymult(), to call cppmult().
def pymult(int_param, float_param):
    # A Python function that has access to the C++ function cppmult.
    return cppmult(int_param, float_param)
