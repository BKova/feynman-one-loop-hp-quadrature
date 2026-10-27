from dataclasses import dataclass

import numpy as np

from .base import SetupBase

@dataclass
class BoxParams:
    alpha: np.array
    beta: np.array
    gamma: np.array
    m1: float
    m2: float
    m3: float
    m4: float
    reference_solution: float

class SetupBox(SetupBase) :
    def __init__(self, problem):
        self.__problem = problem

    def get_parameters(self):
        if self.__problem == "box1":
            return BoxParams(
                np.array([ 31.5487, -322.403, 300.53, -385.58 ], dtype=float),
                np.array([ 103.904, 202.01, -451.278, -435.128 ], dtype=float),
                np.array([ 294.767, 252.89, 447.092, 311.716 ], dtype=float),
                4.68481,4.68481,4.68481,4.68481,
                -3.4005008473604458e-11
            )
        elif self.__problem == "box2":
            return BoxParams(
                np.array([ 50.8543, -55.7461, 11.6999, 94.9259 ], dtype=float),
                np.array([ 0.69914, 67.1926, -5.78627, 91.5278 ], dtype=float),
                np.array([ 52.3577, 76.3226, 43.8222, 13.0587], dtype=float),
                54.2965, 53.5406, 55.9681, 51.7444,
                -2.1981438636707011e-9
            )
        elif self.__problem == "box3":
            return BoxParams(
                np.array([ -2, -1, -1, -np.sqrt(2)], dtype=float),
                np.array([ -2, -1, -1, -np.sqrt(2)], dtype=float),
                np.array([ -2, -1, -1, -np.sqrt(2)], dtype=float),
                0.25, 0.5, 1, 0,
                -5.91485594077820
            )
        elif self.__problem == "box4":
            return BoxParams(
                np.array([-np.sqrt(0.6), 0.5/np.sqrt(0.6), 0, 0.1*np.sqrt(11)/np.sqrt(0.6)], dtype=float),
                np.array([np.sqrt(0.6), -np.sqrt(0.6), 0, 0], dtype=float),
                np.array([0, -0.5/np.sqrt(0.6), 0, -0.1*np.sqrt(11)/np.sqrt(0.6)], dtype=float),
                0,0,0,0,
                3.466919141335241e-1
            )
        elif self.__problem == "box5":
            return BoxParams(
                np.array([-np.sqrt(0.9), np.sqrt(0.73)/np.sqrt(0.9), 0, 0.1*np.sqrt(8)/np.sqrt(0.9)], dtype=float),
                np.array([np.sqrt(0.9), -np.sqrt(0.9), 0, 0], dtype=float),
                np.array([0, -np.sqrt(0.73)/np.sqrt(0.9), 0, -0.1*np.sqrt(8)/np.sqrt(0.9)], dtype=float),
                0,0,0,0,
                8.102928427986452e-1
            )
        elif self.__problem == "box6":
            return BoxParams(
                np.array([ -1, 0.4,0,np.sqrt(0.84)], dtype=float),
                np.array([ 1, -1, 0, 0], dtype=float),
                np.array([0, -0.4,0,-np.sqrt(0.84)], dtype=float),
                0,0,0,0,
                -1.453245298343436e-2
            )

    def derive_singular_points(self):
        params = self.get_parameters()
        alpha = params.alpha
        beta = params.beta
        gamma = params.gamma
        return np.array([
            [0,0,0],
            -alpha[1:],
            (-alpha-beta)[1:],
            (-alpha-beta-gamma)[1:]
        ])

    def get_max_refinement_level_practical(self):
        if self.__problem == "box4":
            return 52
        elif self.__problem == "box5":
            return 52
        elif self.__problem == "box6":
            return 55
        return 40

    def derive_mesh_parameter_b(self):
        points = self.derive_singular_points()
        furthest_point_radius = np.max(np.linalg.norm(points,axis=1))

        return np.max([1, 2 * furthest_point_radius * (1 + np.sqrt(3))/np.sqrt(3)])

    def derive_mesh_parameter_b_practical(self):
        # scale by 40 is done for no other reason than to numerically reduce errors in the transforms
        # by not transforming until less significant values, we can push precision from 1e-14 to 1e-16
        # alternatively sin/cos transforms can be rewriten for higher precision
        return 20 * self.derive_mesh_parameter_b()