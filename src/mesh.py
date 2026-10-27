import numpy as np
from collections import deque
from mpmath import mp
from .helpers import compute_addmisibility_helper, dyadic_partition_helper, safe_numeric_to_numpy


class DyadicMesh:
    def __init__(self, lp, ext):
        self.__dim = len(lp)
        self.__lp = lp
        self.__lp_np = safe_numeric_to_numpy(lp)
        self.__ext = ext
        self.__ext_np = safe_numeric_to_numpy(ext)
        self.__unprocessed = deque([(self.__lp_np, self.__ext_np)])
        self.__processed = []
        self.__ref_points = []

    def add_singular_point(self, point, axes=None):
        if axes == None:
            axes = np.arange(self.__dim)
        self.__ref_points.append((safe_numeric_to_numpy(point), axes))

    def refine(self, absolute_max_refinement_orders, relative=False):
        max_refinements = self.__ext_np*np.power(2.0, -absolute_max_refinement_orders)
        while len(self.__unprocessed) > 0: 
            lp, ext = self.__unprocessed.pop()
            partition_axes = set()
            for point, axes in self.__ref_points:
                addmissible = compute_addmisibility_helper(lp, ext, point, axes, relative)
                if not addmissible:
                    partition_axes.update(axes)

            if len(partition_axes) == 0:
                self.__processed.append([lp, ext])
            else:
                partitioned_elems, next_ext = dyadic_partition_helper(lp, ext, partition_axes, max_refinements)
                if len(partitioned_elems) > 1:
                    for el in partitioned_elems:
                        self.__unprocessed.append((el, next_ext))

    def get_processed(self):
        processed_np = np.array(self.__processed)
        return processed_np[:, 0, :], processed_np[:,1,:]

    def get_high_precision_processed(self):
        lps, exts = self.get_processed()

        # Obtain exact integer coefficients
        ext_orders = np.log2(self.__ext_np/exts).round().round().astype(int)
        lps_relative_order = ((lps-self.__lp_np)/exts).round().astype(int)
        # Exact integers can give convert refinement to higher precision one
        exts = [[
            self.__ext[j]*mp.power(2,-ext_orders[i][j]) for j in range(len(self.__lp))
        ] for i in range(len(lps))]
        lps = [[
            self.__lp[j] + exts[i][j] * lps_relative_order[i][j] for j in range(len(self.__lp))
        ] for i in range(len(lps))]

        return lps, exts

    
    def get_degs(self, M, alpha, beta):
        _, exts = self.get_processed()
        reduction =  np.log2(self.__ext_np / exts)
        level = M - reduction # M-m
        deg = alpha + np.ceil(beta*level).round().astype(int)
        deg = deg * (deg >= 1) + (deg < 1) # ensure at least degree 1
        return deg
