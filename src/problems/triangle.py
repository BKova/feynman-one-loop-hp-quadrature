from typing import override
from abc import ABC, abstractmethod

import numpy as np
from mpmath import mp, mpf
from xsum import xsum_small_accumulator, xsum_add

from ..helpers import break_and_return_xsum_helper

class Triangle:
    def __init__(self, params):
        self.__alpha = params.alpha #p1
        self.__beta = params.beta   #p2
        self.__m1 = params.m1
        self.__m2 = params.m2
        self.__m3 = params.m3

    def evaluate_unrolled(self, coords, weights):
        M1 = self.__m1
        M2 = self.__m2
        M3 = self.__m3
        ALPHA_0 = self.__alpha[0]
        BETA_0 = self.__beta[0]
        ALPHA = self.__alpha[1:]
        BETA = self.__beta[1:]
        coords_trunc = coords[:,:3]
        OFFSET = coords[:,3] if coords.shape[1] == 4 else 0

        T = OFFSET*(OFFSET-2)/np.power(OFFSET - 1,2)

        E1 = np.sqrt(np.sum((coords_trunc + ALPHA)**2,axis=1)+M1**2 - T)
        E2 = np.sqrt(np.sum((coords_trunc + ALPHA + BETA)**2,axis=1)+M2**2 - T)
        E3 = np.sqrt(np.sum(coords_trunc**2,axis=1)+M3**2 - T)
        E12 = E1 + E2 + BETA_0
        E21 = E1 + E2 - BETA_0
        E13 = E1 + E3 + ALPHA_0
        E31 = E1 + E3 - ALPHA_0
        E23 = E2 + E3 + ALPHA_0 + BETA_0
        E32 = E2 + E3 - ALPHA_0 - BETA_0
        integrand_scale = weights/(4*np.pi*E1*E2*E3)

        return np.array([
            integrand_scale/(E31*E32),
            integrand_scale/(E21*E32),
            integrand_scale/(E12*E31),
            integrand_scale/(E13*E23),
            integrand_scale/(E12*E23),
            integrand_scale/(E21*E13)
        ])

    def evaluate(self, coords, weights):
        accumulator = xsum_small_accumulator()
        xsum_add(accumulator, self.evaluate_unrolled(coords, weights).ravel())
        # xsum does not support serializtion, so we decompose it
        return break_and_return_xsum_helper(accumulator)



# Used for high precision mpmath (non tensorized) parameters
class HighPrecisionTriangle:
    def __init__(self, params):
        self.__alpha = params.alpha
        self.__beta = params.beta
        self.__m1 = params.m1
        self.__m2 = params.m2
        self.__m3 = params.m3

    def evaluate(self, coords, weights):
        accumulator = mpf("0")
        for coord, weight in zip(coords, weights):
            M1 = self.__m1
            M2 = self.__m2
            M3 = self.__m3
            ALPHA_0 = self.__alpha[0]
            BETA_0 = self.__beta[0]
            ALPHA = self.__alpha[1:]
            BETA = self.__beta[1:]

            E1 = mp.sqrt(
                (coord[0] + ALPHA[0])**2 +
                (coord[1] + ALPHA[1])**2 +
                (coord[2] + ALPHA[2])**2 + M1**2)
            E2 = mp.sqrt(
                (coord[0] + ALPHA[0] + BETA[0])**2 +
                (coord[1] + ALPHA[1] + BETA[1])**2 +
                (coord[2] + ALPHA[2] + BETA[2])**2 + M2**2)
            E3 = mp.sqrt(
                (coord[0])**2 +
                (coord[1])**2 +
                (coord[2])**2 + M3**2)

            E12 = E1 + E2 + BETA_0
            E21 = E1 + E2 - BETA_0
            E13 = E1 + E3 + ALPHA_0
            E31 = E1 + E3 - ALPHA_0
            E23 = E2 + E3 + ALPHA_0 + BETA_0
            E32 = E2 + E3 - ALPHA_0 - BETA_0

            integrand_scale = weight/(4*mp.pi*E1*E2*E3)
            accumulator += integrand_scale * (
                1/(E31*E32) + 1/(E21*E32) + 1/(E12*E31) +
                1/(E13*E23) + 1/(E12*E23) + 1/(E21*E13)
            )
        return [accumulator]

class HigherOrderCoeffBase(ABC):
    def __init__(self, params):
        self.__triangle = Triangle(params)

    @abstractmethod
    def compute_base_shift(self, t):
        return NotImplemented

    @abstractmethod
    def compute_part_shift(self, t):
        return NotImplemented

    @abstractmethod
    def compute_inv_part_shift(self, t):
        return NotImplemented

    def evaluate(self, coords, weights):
        t = coords[:,3]
        base_shift = self.compute_base_shift(t)
        part_shift = self.compute_part_shift(t)
        inv_part_shift = self.compute_inv_part_shift(t)

        base = self.__triangle.evaluate_unrolled(np.hstack([coords[:,:-1], np.zeros(len(t))[:,None]]), weights)
        part = self.__triangle.evaluate_unrolled(coords, weights)
        inv_part = self.__triangle.evaluate_unrolled(np.hstack([coords[:,:-1], 1.0-t[:,None]]), weights)

        accumulator = xsum_small_accumulator()
        xsum_add(accumulator, (base_shift * base / t).ravel())
        xsum_add(accumulator, (part_shift * part / t).ravel())
        xsum_add(accumulator, (inv_part_shift * inv_part / t).ravel())

        return break_and_return_xsum_helper(accumulator)
    

class HigherOrderCoeffProblemE1(HigherOrderCoeffBase):
    def __init__(self, params):
        super().__init__(params)

    @override
    def compute_base_shift(self, t):
        return -(-1 + np.euler_gamma * t  + t*np.log(2))

    @override
    def compute_part_shift(self, t):
        return -(2*(t - 1))/(t - 2)

    @override
    def compute_inv_part_shift(self, t):
        return -2.0

class HigherOrderCoeffProblemE2(HigherOrderCoeffBase):
    def __init__(self, params):
        super().__init__(params)
    @override
    def compute_base_shift(self, t):
        return (
                    6 * np.euler_gamma**2 * t
                    - np.pi**2 * t
                    - 12 * np.log(2)
                    + 6 * t * np.log(2)**2
                    + 12 * np.euler_gamma * (-1 + t * np.log(2))
                    - 12 * np.log(t)
                ) / (12.0)

    @override
    def compute_part_shift(self, t):
        return (
                    -2 * np.euler_gamma
                    + 2 * np.euler_gamma * t
                    - (-2 + t) * np.log(-((-1 + t)**2 / (-2 + t)))
                    - t * np.log(-((-1 + t)**2 / ((-2 + t) * t)))
                    - 2 * np.log(t)
                    + t * np.log(t)
                ) / ((-2 + t))

    @override
    def compute_inv_part_shift(self, t):
        return 2*(np.euler_gamma - np.log((t**2 / (1 - t**2))))
