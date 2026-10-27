import math
import numpy as np
from src.setups.triangle import SetupTriangle
from src.problems.triangle import HigherOrderCoeffProblemE1, HigherOrderCoeffProblemE2
from src.evaluate import Evaluation
from src.transforms import CubeSphereGapTransform, ResidualTransform, TruncatedGapTransform, TruncatedResidualTransform
from src.mesh import DyadicMesh

def run_higher_order_coeffs_example(cpu_count):
    setup = SetupTriangle("triangle8")
    params = setup.get_parameters()
    singular_points = setup.derive_singular_points()
    mesh_b = setup.derive_mesh_parameter_b_practical()
    # Going under seriously diminishes precision
    # as contributions are significant on the edges.
    M = 50
    alpha = 1
    beta =  0.3
    evaluation = Evaluation()

    gap_quadrature_node_count = 40
    residual_quadrature_node_count = 40

    center_mesh = DyadicMesh(
        np.array([-mesh_b, -mesh_b, -mesh_b, 0]),
        np.array([2 * mesh_b, 2 * mesh_b, 2 * mesh_b, 1])
    )
    center_mesh.add_singular_point(np.array([0]), axes=[3])
    center_mesh.add_singular_point(np.array([1]), axes=[3])
    for p in singular_points:
        center_mesh.add_singular_point(p, axes=[0,1,2])
    center_mesh.refine(M)
    lps, exts = center_mesh.get_processed()
    degs = center_mesh.get_degs(M, alpha, beta)

    gap_mesh = DyadicMesh(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0])
    )
    gap_mesh.add_singular_point(np.array([0]), axes=[3])
    gap_mesh.add_singular_point(np.array([1]), axes=[3])
    gap_mesh.refine(M)
    gap_lps, gap_exts = gap_mesh.get_processed()
    gap_degs = gap_mesh.get_degs(M, alpha, beta)
    # adjust regular orders
    gap_degs[:, :-1] = np.ones(gap_degs[:, :-1].shape) * gap_quadrature_node_count

    residual_mesh = DyadicMesh(
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, np.pi, 2*np.pi, 1.0])
    )
    residual_mesh.add_singular_point(np.array([0]), axes=[3])
    residual_mesh.add_singular_point(np.array([1]), axes=[3])
    residual_mesh.refine(M)
    residual_lps, residual_exts = residual_mesh.get_processed()
    residual_degs = residual_mesh.get_degs(M, alpha, beta)
    # adjust regular orders
    residual_degs[:, :-1] = np.ones(residual_degs[:, :-1].shape) * residual_quadrature_node_count

    problemE1 = HigherOrderCoeffProblemE1(params)
    problemE2 = HigherOrderCoeffProblemE2(params)

    transform_A = TruncatedGapTransform(CubeSphereGapTransform(mesh_b, 2*mesh_b))
    transform_B = TruncatedResidualTransform(ResidualTransform(2*np.sqrt(3)*mesh_b))

    execution_batchesE1 = [
        (divmod(len(lps), 100)[0], cpu_count, lps, exts, degs, [], problemE1, 1.0),
        (divmod(len(gap_lps), 100)[0], cpu_count, gap_lps, gap_exts, gap_degs, [transform_A], problemE1, 1.0),
        (divmod(len(residual_lps), 100)[0], cpu_count, residual_lps, residual_exts, residual_degs, [transform_B], problemE1, 1.0),
    ]

    execution_batchesE2 = [
        (divmod(len(lps), 100)[0], cpu_count, lps, exts, degs, [], problemE2, 1.0),
        (divmod(len(gap_lps), 100)[0], cpu_count, gap_lps, gap_exts, gap_degs, [transform_A], problemE2, 1.0),
        (divmod(len(residual_lps), 100)[0], cpu_count, residual_lps, residual_exts, residual_degs, [transform_B], problemE2, 1.0),
    ]

    result_generatorE1 = evaluation.generate_evaluation_executions(execution_batchesE1)
    result_generatorE2 = evaluation.generate_evaluation_executions(execution_batchesE2)

    # We halve the total evaluation to give evaluations per problem
    return math.fsum(result_generatorE1), math.fsum(result_generatorE2) , evaluation.evaluation_counter/2