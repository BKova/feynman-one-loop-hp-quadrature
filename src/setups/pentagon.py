from dataclasses import dataclass

import numpy as np

from .base import SetupBase

@dataclass
class PentagonParams:
    alpha: np.array
    beta: np.array
    gamma: np.array
    delta: np.array
    m1: float
    m2: float
    m3: float
    m4: float
    m5: float
    reference_solution: float

class SetupPentagon(SetupBase) :
    def __init__(self, problem):
        self.__problem = problem

    def get_parameters(self):
        if self.__problem == "pentagon1":
            return PentagonParams(
                np.array([ 33.7452, 45.7273, 31.1525, -7.47943 ], dtype=float),
                np.array([ 31.3644, -41.5073, 46.479, 2.04203 ], dtype=float),
                np.array([ 4.59005, 17.0701, 32.654, 41.9363 ], dtype=float),
                np.array([ 29.5105, -28.2596, 46.1733, -35.0892 ], dtype=float),
                5.01213,5.01213,5.01213,5.01213,5.01213,
                1.9585377709960810e-11
            )
        elif self.__problem == "pentagon2":
            return PentagonParams(
                np.array([ 33.7648, 45.4406, -10.6808, 16.4193 ], dtype=float),
                np.array([ 72.935, 67.4917, -11.8149, -36.2846 ], dtype=float),
                np.array([ 8.01673, -49.4011, -66.092, -0.11414 ], dtype=float),
                np.array([ -86.5419, -97.0123, 68.1249, 32.9488 ], dtype=float),
                98.427,28.8906,40.5144,75.4564,11.0833,
                2.3427532470686740e-12
            )

    def derive_singular_points(self):
        params = self.get_parameters()
        alpha = params.alpha
        beta = params.beta
        gamma = params.gamma
        delta = params.delta
        return np.array([
            [0,0,0],
            -alpha[1:],
            (-alpha-beta)[1:],
            (-alpha-beta-gamma)[1:],
            (-alpha-beta-gamma-delta)[1:]
        ])
    
    def get_max_refinement_level_practical(self):
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