import sys
from src.setups.triangle import SetupTriangle
from src.setups.box import SetupBox
from src.setups.pentagon import SetupPentagon
from src.problems.triangle import Triangle
from src.problems.box import Box, HighPrecisionIntegrands
from src.problems.pentagon import Pentagon
from regular_example import run_regular_example
from numerator_example import run_numerator_example
from threshold_example import run_threshold_example
from higher_order_coeffs_example import run_higher_order_coeffs_example
from mpmath import mp


# set precision
mp.dps = 50
cpu_count = int(sys.argv[1])
box_numerator_lib = "" if len(sys.argv) < 3 else sys.argv[2]


def run_regular(label, name, setup_class, problem_class):
    print(f"Problem: {label}")
    setup = setup_class(name)
    problem = problem_class(setup.get_parameters())
    result, reference, eval_count = run_regular_example(setup, problem, cpu_count, practical=True)
    print(f'Result: {format(result, '.16e')}')
    print(f'Refer.: {format(reference, '.16e')}')
    print(f'Total number of evaluations: {eval_count}')

def run_numerator(label, name):
    print(f"Problem: {label}", flush=True)
    setup = SetupBox(name)
    problem = HighPrecisionIntegrands(box_numerator_lib, name)
    result, reference, eval_count = run_numerator_example(setup, problem, cpu_count)
    print(f'Result: {format(result, '.16e')}', flush=True)
    print(f'Refer.: {format(reference, '.16e')}', flush=True)
    print(f'Total number of evaluations: {eval_count}', flush=True)

def run_threshold(label, name):
    print(f"Problem: {label}")
    setup = SetupTriangle(name)
    output = run_threshold_example(setup, cpu_count)
    print(f'Result: {format(output[0], '.16e')}')
    print(f'Refer.: {format(output[2], '.16e')}')
    print(f'Imagi.: {format(output[1], '.16e')}')
    print(f'Im.Ref: {format(output[3], '.16e')}')
    print(f'Total number of evaluations: {output[4]}')

def run_higher_order_coeff():
    print("Problem: Higher order coeffs", flush=True)
    order_1_reference = -5.7349778469765868e-2
    order_2_reference = 1.270712099300321e-1
    order_1, order_2, eval_count = run_higher_order_coeffs_example(cpu_count)
    print(f'Order 1: {format(order_1, '.16e')}', flush=True)
    print(f'Ref. O1: {format(order_1_reference, '.16e')}', flush=True)
    print(f'Order 2: {format(order_2, '.16e')}', flush=True)
    print(f'Ref. O2: {format(order_2_reference, '.16e')}', flush=True)
    print(f'Total number of evaluations per order: {eval_count}', flush=True)

run_regular("Triangle 1", "triangle1", SetupTriangle, Triangle)
run_regular("Triangle 2", "triangle2", SetupTriangle, Triangle)
run_regular("Triangle 3", "triangle3", SetupTriangle, Triangle)
run_regular("Triangle 4", "triangle4", SetupTriangle, Triangle)
run_regular("Box 1", "box1", SetupBox, Box)
run_regular("Box 2", "box2", SetupBox, Box)
run_regular("Box 3", "box3", SetupBox, Box)
run_regular("Pentagon 1", "pentagon1", SetupPentagon, Pentagon)
run_regular("Pentagon 2", "pentagon2", SetupPentagon, Pentagon)
run_threshold("Triangle 5", "triangle5")
run_threshold("Triangle 6", "triangle6")
run_threshold("Triangle 7", "triangle7")
if box_numerator_lib:
    run_numerator("Box 4", "box4")
    run_numerator("Box 5", "box5")
    run_numerator("Box 6", "box6")
run_higher_order_coeff()