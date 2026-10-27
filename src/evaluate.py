import numpy as np
from multiprocessing import Pool
from mpmath import mp
from .quadrature import element_gauss_legendre
from .helpers import partition_by_time_helper, safer_sum_helper
from xsum import xsum_small_accumulator

def compute_problem_mp(params):
    lps, exts, degs, transforms, problem, scale = params
    # set multipreceision on new process
    mp.dps = 50

    coords, weights= element_gauss_legendre(lps, exts, degs, scale)
    for T in transforms:
        coords, weights = T.transform(coords, weights)
    return problem.evaluate(coords, weights)

class Evaluation:
    def __init__(self):
        self.evaluation_counter = 0

    def generate_evaluation_executions(self, execution_batches):
        for execution_batch in execution_batches:
            chunks, num_processes, lps, exts, degs, transforms, problem, scale = execution_batch
            # if we are computing high precision (no numpy), don't collapse result
            running_high_precision = not isinstance(lps, np.ndarray)

            self.evaluation_counter += np.sum(np.prod(degs,axis=1)).round().astype(int)
            evaluation_numbers = np.prod(degs, axis=1)
            groups, _ = partition_by_time_helper(np.arange(len(lps)), evaluation_numbers, chunks + 1)

            work_chunks = None
            if running_high_precision:
                work_chunks = [(
                    np.array(lps)[groups[i]].tolist(),
                    np.array(exts)[groups[i]].tolist(),
                    np.array(degs)[groups[i]].tolist(),
                    transforms, problem, scale
                    ) for i in range(chunks + 1)]
            else:
                work_chunks = [(
                    lps[groups[i]], exts[groups[i]], degs[groups[i]],
                    transforms, problem, scale
                ) for i in range(chunks + 1)]

            if num_processes > 1:
                with Pool(num_processes) as pool:
                    results = pool.imap(compute_problem_mp, work_chunks)
                    for res in results:
                        yield from res
            else:
                for chunk in work_chunks:
                    yield from compute_problem_mp(chunk)