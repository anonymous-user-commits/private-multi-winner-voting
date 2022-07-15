# Run cython example

Install the following packages:
`python3 -m pip install invoke cython pybind11`

Then run:
`invoke all`

The expected output:
```
$ invoke all
Building C++ Library
Build cppmult complete.
Building Cython Module
Compile and link the cython wrapper library.
Python module compiled.
Building Cython Module completed.
Testing Cython Module
    In cppmul: int 6 float 2.3 returning  13.8
    In Python: int: 6 float 2.3 return val 13.8
```