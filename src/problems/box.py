import numpy as np
from xsum import xsum_small_accumulator, xsum_add

from ..high_prec_bindings import HighPrecisionIntegrationBindings
from ..helpers import break_and_return_xsum_helper

class Box:
    def __init__(self, params):
        self.__alpha = params.alpha #p1
        self.__beta = params.beta   #p2
        self.__gamma = params.gamma   #p3
        self.__m1 = params.m1
        self.__m2 = params.m2
        self.__m3 = params.m3
        self.__m4 = params.m4

    def evaluate(self, coords, weights):
        M1 = self.__m1
        M2 = self.__m2
        M3 = self.__m3
        M4 = self.__m4
        ALPHA_0 = self.__alpha[0]
        BETA_0 = self.__beta[0]
        GAMMA_0 = self.__gamma[0]
        ALPHA = self.__alpha[1:]
        BETA = self.__beta[1:]
        GAMMA = self.__gamma[1:]

        E1 = np.sqrt(np.sum((coords + ALPHA)**2,axis=1)+M1**2)
        E2 = np.sqrt(np.sum((coords + ALPHA + BETA)**2,axis=1)+M2**2)
        E3 = np.sqrt(np.sum((coords + ALPHA + BETA + GAMMA)**2,axis=1)+M3**2)
        E4 = np.sqrt(np.sum(coords**2,axis=1)+M4**2)

        E1p1 = E1 - ALPHA_0
        E1p2 = E1 + BETA_0
        E2p2 = E2 - BETA_0
        E2p3 = E2 + GAMMA_0
        E3p3 = E3 - GAMMA_0
        E4p1 = E4 + ALPHA_0
        E1p23 = E1p2 + GAMMA_0
        E2p12 = E2p2 - ALPHA_0
        E3p23 = E3p3 - BETA_0
        E4p12 = E4p1 + BETA_0
        E3p123 = E3p23 - ALPHA_0
        E4p123 = E4p1 + BETA_0 + GAMMA_0

        accumulator = xsum_small_accumulator()

        integrand_scale = -weights / (8.*E1*E2*E3*E4*np.pi)
        # components are passed individually in order reduce truncation errors
        xsum_add(accumulator, np.array([
            integrand_scale / ((E1p2 + E2)*(E1p23 + E3)*(E1p1 + E4)),
            integrand_scale / ((E1p23 + E3)*(E2p3 + E3)*(E1p1 + E4)),
            integrand_scale / ((E1p2 + E2)*(E2 + E3p3)*(E1p1 + E4)),
            integrand_scale / ((E1 + E2p2)*(E2p3 + E3)*(E2p12 + E4)),
            integrand_scale / ((E1 + E2p2)*(E1 + E3p23)*(E2p12 + E4)),
            integrand_scale / ((E2p3 + E3)*(E1p1 + E4)*(E2p12 + E4)),
            integrand_scale / ((E1 + E3p23)*(E2 + E3p3)*(E3p123 + E4)),
            integrand_scale / ((E2 + E3p3)*(E1p1 + E4)*(E3p123 + E4)),
            integrand_scale / ((E1 + E3p23)*(E2p12 + E4)*(E3p123 + E4)),
            integrand_scale / ((E1p1 + E4)*(E2p12 + E4)*(E3p123 + E4)),
            integrand_scale / ((E1 + E2p2)*(E2p3 + E3)*(E1 + E4p1)),
            integrand_scale / ((E1 + E2p2)*(E1 + E3p23)*(E1 + E4p1)),
            integrand_scale / ((E1 + E3p23)*(E2 + E3p3)*(E1 + E4p1)),
            integrand_scale / ((E1p2 + E2)*(E1p23 + E3)*(E2 + E4p12)),
            integrand_scale / ((E1p2 + E2)*(E2 + E3p3)*(E2 + E4p12)),
            integrand_scale / ((E2 + E3p3)*(E1 + E4p1)*(E2 + E4p12)),
            integrand_scale / ((E1p23 + E3)*(E2p3 + E3)*(E3 + E4p123)),
            integrand_scale / ((E2p3 + E3)*(E1 + E4p1)*(E3 + E4p123)),
            integrand_scale / ((E1p23 + E3)*(E2 + E4p12)*(E3 + E4p123)),
            integrand_scale / ((E1 + E4p1)*(E2 + E4p12)*(E3 + E4p123))
        ]).ravel())

        return break_and_return_xsum_helper(accumulator)

class HighPrecisionIntegrands:
    def __init__(self, library_location, problem):
        self.__library_location = library_location
        self.__problem_index = -1
        if problem == "triangle5":
            self.__problem_index = 1
        elif problem == "triangle6":
            self.__problem_index = 2
        elif problem == "triangle7":
            self.__problem_index = 3
        elif problem == "box4":
            self.__problem_index = 4
        elif problem == "box5":
            self.__problem_index = 5
        elif problem == "box6":
            self.__problem_index = 6

    def evaluate(self, coords, weights):
        # This has to be initialized on each process
        lib_handle = HighPrecisionIntegrationBindings(self.__library_location, self.__problem_index)
        return np.array(lib_handle.integrate(coords.flatten(), weights)).ravel()