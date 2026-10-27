import numpy as np
import math
from mpmath import mp, mpf
from src.transforms import SphericalToCarthesianTransform, ResidualTransform, TrippleTiledSurfaceTransform, ThresholdImaginaryTransform
from src.mesh import DyadicMesh
from src.evaluate import Evaluation
from src.problems.triangle import Triangle, HighPrecisionTriangle
from src.geometry_helpers import derive_angular_singularities_params

def run_threshold_example(setup, cpu_count):
    params = setup.get_parameters()
    prec_params = setup.get_high_precision_params()
    singular_points = setup.derive_singular_points()
    M = setup.get_max_refinement_level_practical()
    alpha = setup.get_quad_scaling_alpha()
    beta =  setup.get_quad_scaling_beta()
    residual_quadrature_node_count = 50
    threshold_quadrature_node_count = 50
    threshold_refinement_levels = 60
    evaluation = Evaluation()
    problem = Triangle(params)
    problem_high_prec = HighPrecisionTriangle(prec_params)

    integration_center = -params.alpha[1:]
    integration_center_prec = [-p for p in prec_params.alpha[1:]]

    # Unintegrable surface is at that radius
    ellip_radius = np.sqrt(np.power(params.alpha[0],2)/4-np.power(params.m1,2))
    ellip_radius_mpf = mp.sqrt(mp.power(prec_params.alpha[0],2)/4-mp.power(prec_params.m1,2))
    # We split integration domains into: |strong singualrities |weak singularities |residual
    first_section_end = 1.5 * ellip_radius
    second_section_end = 2 * np.max(np.linalg.norm(singular_points - integration_center,axis=1))

    # We do mirrored refinement, ensuring that cancellation is done with high precision
    # This mesh is tiled 3x times at the start 0-><-|->, see TrippleTiledSurfaceTransform
    inner_mesh = DyadicMesh([0.0, 0.0, 0.0], [ellip_radius_mpf/2, mp.pi, 2*mp.pi])
    inner_mesh.add_singular_point([mpf("0")], axes=[0])
    inner_mesh.refine(threshold_refinement_levels)

    lps_inner_prec, exts_inner_prec = inner_mesh.get_high_precision_processed()
    degs_inner = inner_mesh.get_degs(threshold_refinement_levels, alpha, beta) # 25 * np.ones(np.array(lps_inner_prec).shape).round().astype(int)

    line_sings_periodic, point_sings_periodic = derive_angular_singularities_params(singular_points, integration_center)
    center_mesh = DyadicMesh(
        np.array([first_section_end, 0, 0]),
        np.array([second_section_end-first_section_end, np.pi, 2*np.pi])
    )
    for point in point_sings_periodic:
        center_mesh.add_singular_point(point)

    for line in line_sings_periodic:
            center_mesh.add_singular_point(line, axes=[1,2])

    center_mesh.add_singular_point(np.array([ellip_radius]), axes=[0])
    center_mesh.refine(M, relative=True)

    lps_center, exts_center = center_mesh.get_processed()
    degs_center = center_mesh.get_degs(M, alpha, beta)

    transform_A = TrippleTiledSurfaceTransform(ellip_radius_mpf)
    transform_B = SphericalToCarthesianTransform(integration_center_prec)
    transform_C = SphericalToCarthesianTransform(integration_center)
    transform_D = ResidualTransform(second_section_end, integration_center)
    transform_E = ThresholdImaginaryTransform(ellip_radius_mpf)

    residual_lps = np.array([[0.0,0.0,0.0]])
    residual_ext = np.array([[1.0, np.pi, 2*np.pi]])
    residual_degs = np.array([np.ones(3)])*residual_quadrature_node_count

    imaginary_lps = [[mpf("0"), mpf("0")]]
    imaginary_ext = [[1*mp.pi, 2*mp.pi]]
    imaginary_degs = [[threshold_quadrature_node_count, threshold_quadrature_node_count]]

    execution_batches_real = [(
        min(cpu_count, len(lps_inner_prec)-1),
        cpu_count, lps_inner_prec, exts_inner_prec, degs_inner,
        [transform_A, transform_B], problem_high_prec, 1.0
        ),(
        divmod(len(lps_center), 100)[0],
        cpu_count, lps_center, exts_center, degs_center,
        [transform_C], problem, 1.0
        ),(
            1, 1, residual_lps, residual_ext, residual_degs,
            [transform_D], problem, 1.0
        )]

    execution_batches_imaginary = [
         (0, 1, imaginary_lps, imaginary_ext, imaginary_degs, [transform_E], problem_high_prec, 1.0)
    ]

    result_real = mp.fsum(
        evaluation.generate_evaluation_executions(execution_batches_real)
    )

    result_imaginary = math.fsum(
        evaluation.generate_evaluation_executions(execution_batches_imaginary)
    )

    return result_real, \
           result_imaginary, params.reference_solution,  \
           params.reference_solution_imaginary, evaluation.evaluation_counter