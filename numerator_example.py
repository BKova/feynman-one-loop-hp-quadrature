import math
import numpy as np
from src.transforms import SphericalToCarthesianJacobianOnlyTransform, ResidualTransform
from src.mesh import DyadicMesh
from src.evaluate import Evaluation
from src.geometry_helpers import derive_angular_singularities_params

def run_numerator_example(setup, problem, cpu_count):
    params = setup.get_parameters()
    singular_points = setup.derive_singular_points()
    # The default practical 40 levels are not viable to push full machine
    # precision. Experimentally 52 levels is sufficient to achieve target
    M = setup.get_max_refinement_level_practical()
    alpha = setup.get_quad_scaling_alpha()
    beta =  setup.get_quad_scaling_beta()
    # We integrate with the larger backoff
    interior_domain_radius = 4*setup.derive_mesh_parameter_b()
    evaluation = Evaluation()

    # Refinement levels on the residual
    residual_mesh_M = 65

    # For numerators, we have additional line singularities between E1 and E2/E3/E4
    # Integration is done in spherical coordinates and as such, all singular points need to be translated
    integration_center = -params.alpha[1:]

    line_sings_periodic, point_sings_periodic = derive_angular_singularities_params(singular_points, integration_center)
    radial_mesh = DyadicMesh(np.zeros(3), np.array([interior_domain_radius, np.pi, 2*np.pi]))
    for point in point_sings_periodic:
        radial_mesh.add_singular_point(point)

    for line in line_sings_periodic:
         radial_mesh.add_singular_point(line, axes=[1,2])

    radial_mesh.add_singular_point(np.array([0.0]), axes=[0])

    radial_mesh.refine(M)
    lps, exts = radial_mesh.get_processed()
    degs = radial_mesh.get_degs(M, alpha, beta)

    # We generate the transform centered at -alpha
    transform_A = SphericalToCarthesianJacobianOnlyTransform()
    transform_B = ResidualTransform(interior_domain_radius, jacobian_only=True)

    # Integrate residual R^d\B(4bsqrt(d))
    residual_mesh = DyadicMesh(np.zeros(3), np.array([1.0, np.pi, 2*np.pi]))
    residual_mesh.add_singular_point([0], axes=[0])
    residual_mesh.add_singular_point([1], axes=[0])
    residual_mesh.refine(residual_mesh_M)
    residual_lps, residual_exts = residual_mesh.get_processed()
    residual_degs = residual_mesh.get_degs(60, alpha, beta)

    execution_batches = [
    (
        divmod(len(lps), 100)[0],
        cpu_count, lps, exts, degs,
        [transform_A], problem, 1.0
    ),
    (
        40, 40, residual_lps, residual_exts,
        residual_degs, [transform_B], problem, 1.0
    )]

    result = math.fsum(
        evaluation.generate_evaluation_executions(execution_batches)
    )

    return result, params.reference_solution, evaluation.evaluation_counter