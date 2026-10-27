import numpy as np

class SetupBase:
    def derive_max_refinement_level_M(self, reference_solution):
        C = 1
        # we need to get 16 digitus under reference, in practice small MC run should estable baseline order
        # these parameters are very tight, in practice even 40 levels is enough for machine precision
        sigma = np.min([np.abs(1e-16*reference_solution), 1e-16])
        C_hat = C*(1+np.sqrt(2))*(4*np.sqrt(3)*self.derive_mesh_parameter_b())**3
        return int(np.ceil((np.log2( C_hat*len(self.derive_singular_points()))-np.log2(sigma)+np.log2(1.5)*3)/(2)))
    
    def get_max_refinement_level_practical(self):
        return 40

    def get_quad_scaling_alpha(self):
        return 1

    def get_quad_scaling_beta(self):
        return 0.5

    def derive_mesh_parameter_b(self):
        points = self.derive_singular_points()
        furthest_point_radius = np.max(np.linalg.norm(points,axis=1))

        return np.max([1, 2 * furthest_point_radius * (1 + np.sqrt(3))/np.sqrt(3)])

    def derive_mesh_parameter_b_practical(self):
        # scale by 40 is done for no other reason than to numerically reduce errors in the transforms
        # by not transforming until less significant values, we can push precision from 1e-14 to 1e-16
        # alternatively sin/cos transforms can be rewriten for higher precision
        return 20 * self.derive_mesh_parameter_b()