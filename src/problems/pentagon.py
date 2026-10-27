import numpy as np
from xsum import xsum_small_accumulator, xsum_add

from ..helpers import break_and_return_xsum_helper

class Pentagon:
    def __init__(self, params):
        self.__alpha = params.alpha #p1
        self.__beta = params.beta   #p2
        self.__gamma = params.gamma   #p3
        self.__delta = params.delta   #p4
        self.__m1 = params.m1
        self.__m2 = params.m2
        self.__m3 = params.m3
        self.__m4 = params.m4
        self.__m5 = params.m5

    def evaluate(self, coords, weights):
        M1 = self.__m1
        M2 = self.__m2
        M3 = self.__m3
        M4 = self.__m4
        M5 = self.__m5
        ALPHA_0 = self.__alpha[0]
        BETA_0 = self.__beta[0]
        GAMMA_0 = self.__gamma[0]
        DELTA_0 = self.__delta[0]
        ALPHA = self.__alpha[1:]
        BETA = self.__beta[1:]
        GAMMA = self.__gamma[1:]
        DELTA = self.__delta[1:]

        E1 = np.sqrt(np.sum((coords + ALPHA)**2,axis=1)+M1**2)
        E2 = np.sqrt(np.sum((coords + ALPHA + BETA)**2,axis=1)+M2**2)
        E3 = np.sqrt(np.sum((coords + ALPHA + BETA + GAMMA)**2,axis=1)+M3**2)
        E4 = np.sqrt(np.sum((coords + ALPHA + BETA + GAMMA + DELTA)**2,axis=1)+M4**2)
        E5 = np.sqrt(np.sum(coords**2,axis=1)+M5**2)

        integrand_scale = weights / (16.*E1*E2*E3*E4*E5*np.pi)

        E1p1=E1 - ALPHA_0
        E1p2=E1 + BETA_0
        E2p2=E2 - BETA_0
        E2p3=E2 + GAMMA_0
        E3p3=E3 - GAMMA_0
        E3p4=E3 + DELTA_0
        E4p4=E4 - DELTA_0
        E5p1=E5 + ALPHA_0
        E4p34=E4p4 - GAMMA_0
        E1p23=E1p2 + GAMMA_0
        E1p234=E1p23 + DELTA_0
        E2p34=E2p3 + DELTA_0
        E3p23=E3p3 - BETA_0
        E5p12=E5p1 + BETA_0
        E4p234=E4p34 - BETA_0
        E5p123=E5p12 + GAMMA_0
        E3p123=E3p23 - ALPHA_0
        E2p12=E2p2 - ALPHA_0
        E5p1234=E5p123 + DELTA_0

        accumulator = xsum_small_accumulator()

        xsum_add(accumulator, np.array([
            integrand_scale/((E1p2 + E2)*(E1p23 + E3)*(E1p234 + E4)*(E1p1 + E5)),
            integrand_scale/((E1p23 + E3)*(E2p3 + E3)*(E1p234 + E4)*(E1p1 + E5)),
            integrand_scale/((E1p2 + E2)*(E2 + E3p3)*(E1p234 + E4)*(E1p1 + E5)),
            integrand_scale/((E2p3 + E3)*(E1p234 + E4)*(E2p34 + E4)*(E1p1 + E5)),
            integrand_scale/((E2 + E3p3)*(E1p234 + E4)*(E3p4 + E4)*(E1p1 + E5)),
            integrand_scale/((E1p234 + E4)*(E2p34 + E4)*(E3p4 + E4)*(E1p1 + E5)),
            integrand_scale/((E1p2 + E2)*(E1p23 + E3)*(E2 + E4p34)*(E1p1 + E5)),
            integrand_scale/((E1p2 + E2)*(E2 + E3p3)*(E2 + E4p34)*(E1p1 + E5)),
            integrand_scale/((E1p23 + E3)*(E2p3 + E3)*(E3 + E4p4)*(E1p1 + E5)),
            integrand_scale/((E1p23 + E3)*(E2 + E4p34)*(E3 + E4p4)*(E1p1 + E5)),
            integrand_scale/((E1 + E2p2)*(E2p3 + E3)*(E2p34 + E4)*(E2p12 + E5)),
            integrand_scale/((E1 + E2p2)*(E1 + E3p23)*(E2p34 + E4)*(E2p12 + E5)),
            integrand_scale/((E1 + E3p23)*(E2p34 + E4)*(E3p4 + E4)*(E2p12 + E5)),
            integrand_scale/((E1 + E2p2)*(E2p3 + E3)*(E1 + E4p234)*(E2p12 + E5)),
            integrand_scale/((E1 + E2p2)*(E1 + E3p23)*(E1 + E4p234)*(E2p12 + E5)),
            integrand_scale/((E2p3 + E3)*(E1 + E4p234)*(E3 + E4p4)*(E2p12 + E5)),
            integrand_scale/((E2p3 + E3)*(E2p34 + E4)*(E1p1 + E5)*(E2p12 + E5)),
            integrand_scale/((E2p34 + E4)*(E3p4 + E4)*(E1p1 + E5)*(E2p12 + E5)),
            integrand_scale/((E2p3 + E3)*(E3 + E4p4)*(E1p1 + E5)*(E2p12 + E5)),
            integrand_scale/((E1 + E3p23)*(E2 + E3p3)*(E3p4 + E4)*(E3p123 + E5)),
            integrand_scale/((E1 + E3p23)*(E2 + E3p3)*(E1 + E4p234)*(E3p123 + E5)),
            integrand_scale/((E2 + E3p3)*(E1 + E4p234)*(E2 + E4p34)*(E3p123 + E5)),
            integrand_scale/((E2 + E3p3)*(E3p4 + E4)*(E1p1 + E5)*(E3p123 + E5)),
            integrand_scale/((E2 + E3p3)*(E2 + E4p34)*(E1p1 + E5)*(E3p123 + E5)),
            integrand_scale/((E1 + E3p23)*(E3p4 + E4)*(E2p12 + E5)*(E3p123 + E5)),
            integrand_scale/((E1 + E3p23)*(E1 + E4p234)*(E2p12 + E5)*(E3p123 + E5)),
            integrand_scale/((E3p4 + E4)*(E1p1 + E5)*(E2p12 + E5)*(E3p123 + E5)),
            integrand_scale/((E1 + E2p2)*(E2p3 + E3)*(E2p34 + E4)*(E1 + E5p1)),
            integrand_scale/((E1 + E2p2)*(E1 + E3p23)*(E2p34 + E4)*(E1 + E5p1)),
            integrand_scale/((E1 + E3p23)*(E2 + E3p3)*(E3p4 + E4)*(E1 + E5p1)),
            integrand_scale/((E1 + E3p23)*(E2p34 + E4)*(E3p4 + E4)*(E1 + E5p1)),
            integrand_scale/((E1 + E2p2)*(E2p3 + E3)*(E1 + E4p234)*(E1 + E5p1)),
            integrand_scale/((E1 + E2p2)*(E1 + E3p23)*(E1 + E4p234)*(E1 + E5p1)),
            integrand_scale/((E1 + E3p23)*(E2 + E3p3)*(E1 + E4p234)*(E1 + E5p1)),
            integrand_scale/((E2 + E3p3)*(E1 + E4p234)*(E2 + E4p34)*(E1 + E5p1)),
            integrand_scale/((E2p3 + E3)*(E1 + E4p234)*(E3 + E4p4)*(E1 + E5p1)),
            integrand_scale/((E1 + E4p234)*(E2 + E4p34)*(E3 + E4p4)*(E1 + E5p1)),
            integrand_scale/((E1p2 + E2)*(E1p23 + E3)*(E1p234 + E4)*(E2 + E5p12)),
            integrand_scale/((E1p2 + E2)*(E2 + E3p3)*(E1p234 + E4)*(E2 + E5p12)),
            integrand_scale/((E2 + E3p3)*(E1p234 + E4)*(E3p4 + E4)*(E2 + E5p12)),
            integrand_scale/((E1p2 + E2)*(E1p23 + E3)*(E2 + E4p34)*(E2 + E5p12)),
            integrand_scale/((E1p2 + E2)*(E2 + E3p3)*(E2 + E4p34)*(E2 + E5p12)),
            integrand_scale/((E1p23 + E3)*(E2 + E4p34)*(E3 + E4p4)*(E2 + E5p12)),
            integrand_scale/((E2 + E3p3)*(E3p4 + E4)*(E1 + E5p1)*(E2 + E5p12)),
            integrand_scale/((E2 + E3p3)*(E2 + E4p34)*(E1 + E5p1)*(E2 + E5p12)),
            integrand_scale/((E2 + E4p34)*(E3 + E4p4)*(E1 + E5p1)*(E2 + E5p12)),
            integrand_scale/((E1p23 + E3)*(E2p3 + E3)*(E1p234 + E4)*(E3 + E5p123)),
            integrand_scale/((E2p3 + E3)*(E1p234 + E4)*(E2p34 + E4)*(E3 + E5p123)),
            integrand_scale/((E1p23 + E3)*(E2p3 + E3)*(E3 + E4p4)*(E3 + E5p123)),
            integrand_scale/((E2p3 + E3)*(E2p34 + E4)*(E1 + E5p1)*(E3 + E5p123)),
            integrand_scale/((E2p3 + E3)*(E3 + E4p4)*(E1 + E5p1)*(E3 + E5p123)),
            integrand_scale/((E1p23 + E3)*(E1p234 + E4)*(E2 + E5p12)*(E3 + E5p123)),
            integrand_scale/((E1p23 + E3)*(E3 + E4p4)*(E2 + E5p12)*(E3 + E5p123)),
            integrand_scale/((E3 + E4p4)*(E1 + E5p1)*(E2 + E5p12)*(E3 + E5p123)),
            integrand_scale/((E1p234 + E4)*(E2p34 + E4)*(E3p4 + E4)*(E4 + E5p1234)),
            integrand_scale/((E2p34 + E4)*(E3p4 + E4)*(E1 + E5p1)*(E4 + E5p1234)),
            integrand_scale/((E1p234 + E4)*(E3p4 + E4)*(E2 + E5p12)*(E4 + E5p1234)),
            integrand_scale/((E3p4 + E4)*(E1 + E5p1)*(E2 + E5p12)*(E4 + E5p1234)),
            integrand_scale/((E1p234 + E4)*(E2p34 + E4)*(E3 + E5p123)*(E4 + E5p1234)),
            integrand_scale/((E2p34 + E4)*(E1 + E5p1)*(E3 + E5p123)*(E4 + E5p1234)),
            integrand_scale/((E1p234 + E4)*(E2 + E5p12)*(E3 + E5p123)*(E4 + E5p1234)),
            integrand_scale/((E1 + E5p1)*(E2 + E5p12)*(E3 + E5p123)*(E4 + E5p1234)),
            integrand_scale/((E1 + E4p234)*(E2 + E4p34)*(E3 + E4p4)*(E4p234 + E5 - ALPHA_0)),
            integrand_scale/((E2 + E4p34)*(E3 + E4p4)*(E1p1 + E5)*(E4p234 + E5 - ALPHA_0)),
            integrand_scale/((E1 + E4p234)*(E3 + E4p4)*(E2p12 + E5)*(E4p234 + E5 - ALPHA_0)),
            integrand_scale/((E3 + E4p4)*(E1p1 + E5)*(E2p12 + E5)*(E4p234 + E5 - ALPHA_0)),
            integrand_scale/((E1 + E4p234)*(E2 + E4p34)*(E3p123 + E5)*(E4p234 + E5 - ALPHA_0)),
            integrand_scale/((E2 + E4p34)*(E1p1 + E5)*(E3p123 + E5)*(E4p234 + E5 - ALPHA_0)),
            integrand_scale/((E1 + E4p234)*(E2p12 + E5)*(E3p123 + E5)*(E4p234 + E5 - ALPHA_0)),
            integrand_scale/((E1p1 + E5)*(E2p12 + E5)*(E3p123 + E5)*(E4p234 + E5 - ALPHA_0))
        ]).ravel())

        return break_and_return_xsum_helper(accumulator)