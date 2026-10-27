from dataclasses import dataclass

import numpy as np
from mpmath import mpf

from .base import SetupBase

@dataclass
class TriangleParams:
    alpha: np.array
    beta: np.array
    m1: float
    m2: float
    m3: float
    reference_solution: float
    reference_solution_imaginary: float

@dataclass
class TriangleParamsHighPrec:
    alpha: list
    beta: list
    m1: any
    m2: any
    m3: any
    reference_solution: float
    reference_solution_imaginary: float

class SetupTriangle(SetupBase):
    def __init__(self, problem):
        self.__problem = problem

    def get_parameters(self):
        if self.__problem == "triangle1":
            return TriangleParams(
                np.array([ 1.0, 3.0, 4.0, 5.0 ], dtype=float),
                np.array([ 1.0, 6.0, 7.0, 8.0 ], dtype=float),
                0,0,0,
                1.5420989685426682e-2,0
            )
        elif self.__problem == "triangle2":
            return TriangleParams(
                np.array([ 5.23923, -4.18858, 0.74966, -3.05669 ], dtype=float),
                np.array([ 6.99881, -2.93659, 5.03338, 3.87619], dtype=float),
                7.73358,7.73358,7.73358,
                9.2489103207133699e-3,0
            )
        elif self.__problem == "triangle3":
            return TriangleParams(
                np.array([ 13.4225, 58.7948, -73.1186, -91.9502 ], dtype=float),
                np.array([ 81.6593, -68.5217, 8.75578, -95.0535], dtype=float),
                49.9745,86.9249,80.2257,
                5.3636363075423601e-5,0
            )
        elif self.__problem == "triangle4":
            return TriangleParams(
                np.array([ -2.0, 11.0, 5.0, -3.0 ], dtype=float),
                np.array([ 10.0, 15.0, 14.0, 12.0 ], dtype=float),
                11.0,-8.0,-2.0,
                2.8206297586077667e-3,0
            )
        elif self.__problem == "triangle5":
            return TriangleParams(
                np.array([ 44.3894,0,0,0 ], dtype=float),
                np.array([ 0.62982,-17.8442,17.8442,23.6744], dtype=float),
                7.89824,70.8982,7.89824,
                3.984799769602855e-4,
                4.8774154335541111e-4
            )
        elif self.__problem == "triangle6":
            return TriangleParams(
                np.array([ 70,0,0,0 ], dtype=float),
                np.array([ 1,0.2,100,250 ], dtype=float),
                30,40,30,
                7.8949736287057561e-5,
                2.2258948981264375e-5
            )
        elif self.__problem == "triangle7":
            return TriangleParams(
                np.array([ 42.7,0,0,0], dtype=float),
                np.array([ 7.79,7000.97,320.2,320.1 ], dtype=float),
                20.79,52.42,20.79,
                2.7502744387947350e-7,
                1.452270683160722e-8
            )
        elif self.__problem == "triangle8":
            return TriangleParams(
                np.array([ 1,3,4,5], dtype=float),
                np.array([ 1,6,7,8 ], dtype=float),
                0,0,0,
                -5.7349778469765868e-2,
                1.270712099300321e-1
            )

    def get_high_precision_params(self):
        if self.__problem == "triangle5":
            return TriangleParamsHighPrec(
                [mpf("44.3894"),mpf("0"),mpf("0"),mpf("0") ],
                [mpf("0.62982"),mpf("-17.8442"),mpf("17.8442"),mpf("23.6744")],
                mpf("7.89824"),mpf("70.8982"),mpf("7.89824"),
                3.984799769602855e-4,
                4.8774154335541111e-4
            )
        elif self.__problem == "triangle6":
            return TriangleParamsHighPrec(
                [ mpf("70"),mpf("0"),mpf("0"),mpf("0 ")],
                [ mpf("1"),mpf("0.2"),mpf("100"),mpf("250 ")],
                mpf("30"),mpf("40"),mpf("30"),
                7.8949736287057561e-5,
                2.2258948981264375e-5
            )
        elif self.__problem == "triangle7":
            return TriangleParamsHighPrec(
                [ mpf("42.7"),mpf("0"),mpf("0"),mpf("0")],
                [ mpf("7.79"),mpf("7000.97"),mpf("320.2"),mpf("320.1")],
                mpf("20.79"),mpf("52.42"),mpf("20.79"),
                2.7502744387947350e-7,
                1.452270683160722e-8
            )

    def derive_singular_points(self):
        params = self.get_parameters()
        alpha = params.alpha
        beta = params.beta
        return np.array([
            [0,0,0],
            -alpha[1:],
            (-alpha-beta)[1:]
        ])