"""
Task definitions for invoke command line utility for python bindings.
"""
import glob

import invoke
import os
import shutil
import sys

on_win = sys.platform.startswith("win")


@invoke.task
def clean(c=None):
    """Remove any built objects"""
    for file_pattern in (
        "*.o",
        "*.so",
        "*.obj",
        "*.dll",
        "*.exp",
        "*.lib",
        "*.pyd",
        "cython_wrapper.cpp",
    ):
        for file in glob.glob(file_pattern):
            os.remove(file)
    for dir_pattern in "Release":
        for dir in glob.glob(dir_pattern):
            shutil.rmtree(dir)


@invoke.task()
def build_cppmult(c=None):
    """Build the shared library for the sample C++ code"""
    print("Building C++ Library")
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC cppmult.cpp "
        "-o libcppmult.so "
    )
    print("Build cppmult complete.")


def compile_python_module(cpp_name, extension_name):
    invoke.run(
        "g++ -O3 -Wall -Werror -shared -std=c++11 -fPIC "
        "`python3 -m pybind11 --includes` "
        "-I /home/dockuser/code/he-transformer/build/ext_ngraph_tf/src/ext_ngraph_tf/build_cmake/venv-tf-py3/include/python3.6m -I .  "
        "{0} "
        "-o {1}`python3-config --extension-suffix` "
        "-L. -lcppmult -Wl,-rpath,.".format(cpp_name, extension_name)
    )
    print("Python module compiled.")


@invoke.task()
def build_cython(c=None):
    """Build the cython extension module"""
    print("Building Cython Module")
    # Run cython on the pyx file to create a .cpp file.
    # --cplus tells the compiler to generate a C++ file instead of a C file.
    # -3 switches Cython to generate Python 3 syntax instead of Python 2.
    # -o cython_wrapper.cpp specifies the name of the file to generate.
    invoke.run("cython --cplus -3 cython_example.pyx -o cython_wrapper.cpp")

    print("Compile and link the cython wrapper library.")
    compile_python_module("cython_wrapper.cpp", "cython_example")
    print("Building Cython Module completed.")


@invoke.task()
def test_cython(c=None):
    """Run the script to test Cython"""
    print("Testing Cython Module")
    invoke.run("python3 cython_test.py", pty=True)


@invoke.task(
    clean,
    build_cppmult,
    build_cython,
    test_cython,
)
def all(c=None):
    """Build and run all tests.
    First, it builds the cppmult library and then builds the cython module to
    wrap it.
    """
    pass
