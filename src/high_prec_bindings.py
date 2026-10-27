import numpy as np
from ctypes import *

class HighPrecisionIntegrationBindings():
    def __init__(self, library_location, problem_index):
        self.__lib_handle = cdll.LoadLibrary(library_location)
        self.__lib_handle.integrate.restype = None
        self.__lib_handle.integrate.argtypes = [
            c_int, c_int, c_void_p, c_void_p, c_void_p
        ]
        self.__problem = problem_index
    

    def integrate(self, coords, weigts):
        raw_double_list_type = c_double * len(coords)
        coords_raw_list = raw_double_list_type(*coords)
        weights_raw_list = raw_double_list_type(*weigts)
        out = (c_double * 10)()
        self.__lib_handle.integrate(
            c_int(self.__problem),
            c_int(int(len(coords)/3)),
            coords_raw_list,
            weights_raw_list,
            out
        )

        return list(out)