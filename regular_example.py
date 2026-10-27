import numpy as np
import math
from mpmath import mp
from src.transforms import CubeSphereGapTransform, ResidualTransform
from src.mesh import DyadicMesh
from src.evaluate import Evaluation

def run_regular_example(setup, problem, cpu_count, practical=False, high_precision_result=False):
    params = setup.get_parameters()
    singular_points = setup.derive_singular_points()
    mesh_b = setup.derive_mesh_parameter_b_practical()
    M = setup.get_max_refinement_level_practical()
    alpha = setup.get_quad_scaling_alpha()
    beta =  setup.get_quad_scaling_beta()
    evaluation = Evaluation()

    if not practical:
        # Practical one reduces numerical errors on the residual transform, allows 1e-15 precision instead of 1e-14
        M = setup.derive_max_refinement_level_M(params.reference_solution)

    # In practice these are fine, no refinement is done outside of the center
    gap_quadrature_node_count = 60
    residual_quadrature_node_count = 60

    # Generate central mesh and refine until
    center_mesh = DyadicMesh(-np.ones(3)*mesh_b, 2*np.ones(3)*mesh_b)
    for p in singular_points:
        center_mesh.add_singular_point(p)
    center_mesh.refine(M)
    # Generate transforms: firstly gap between interior and residual
    #                      secondly residual
    transform_A = CubeSphereGapTransform(mesh_b, 2*mesh_b)
    transform_B = ResidualTransform(2*np.sqrt(3)*mesh_b)

    # Get refined elements
    lps, exts = center_mesh.get_processed()
    degs = center_mesh.get_degs(M, alpha, beta)

    # Integrate cube to sphere part
    gap_lps = np.array([[0.0,0.0,0.0]])
    gap_ext = np.array([[1.0, 1.0, 1.0]])
    gap_degs = np.array([np.ones(3)])*gap_quadrature_node_count

    # Integrate residual R^d\B(4bsqrt(d))
    residual_lps = np.array([[0.0,0.0,0.0]])
    residual_ext = np.array([[1.0, np.pi, 2*np.pi]])
    residual_degs = np.array([np.ones(3)])*residual_quadrature_node_count

    execution_batches = [
        (divmod(len(lps), 100)[0], cpu_count, lps, exts, degs, [],       problem, 1.0),
        (1, 1, gap_lps, gap_ext, gap_degs, [transform_A],                problem, 1.0),
        (1, 1, residual_lps, residual_ext, residual_degs, [transform_B], problem, 1.0)
    ]

    result_generator = evaluation.generate_evaluation_executions(execution_batches)

        # lazy evaluate all intermediate results without cancelation
    result = (mp.fsum(result_generator) \
    if high_precision_result \
    else math.fsum(result_generator))

    # Return computed solution, reference solution, number of evaluations
    return result, params.reference_solution, evaluation.evaluation_counter