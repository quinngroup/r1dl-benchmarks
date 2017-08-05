import numpy as np
import numpy.linalg as sla
import os
import unittest

from core_numpy import r1dl

class TestCoreNumpy(unittest.TestCase):

    def test_data1_Z(self):
        self._runTest("tests/t1", "Z")

    def test_data1_D(self):
        self._runTest("tests/t1", "D")

    def test_data2_Z(self):
        self._runTest("tests/t2", "Z")

    def test_data2_D(self):
        self._runTest("tests/t2", "D")

    def _runTest(self, path, result, R = 0.2, M = 5, epsilon = 0.01, rtol = 1e-6):
        S = np.loadtxt(os.path.join(path, "S.txt"))
        Ztrue = np.loadtxt(os.path.join(path, "z_groundtruth.txt"))
        Dtrue = np.loadtxt(os.path.join(path, "D_groundtruth.txt"))

        # Run the test.
        D, Z = r1dl(S, R, M, epsilon, 85271)

        # Since the randomization of the R1DL algorithm can potentially
        # (or rather, will likely) permute the rows of D and Z, we'll
        # need to do a brute-force search for matches.
        if result == "Z":
            self._rowPermutations(Ztrue, Z, rtol)
        else:
            self._rowPermutations(Dtrue, D, rtol)

    def _rowPermutations(self, A_true, A_pred, rtol):
        row_mappings = np.zeros(A_true.shape[0], dtype = np.int) - 1
        pred_rows = list(range(A_pred.shape[0]))
        for index, row in enumerate(A_true):
            found_index = -1
            for i, row_index in enumerate(pred_rows):
                # Perform the assertion.
                try:
                    np.testing.assert_allclose(A_pred[row_index], row, rtol = rtol)
                except AssertionError:
                    pass
                else:
                    # It's ok to do this, because we're breaking anyway.
                    found_index = pred_rows.pop(i)
                    break

            # Did we find a matching row?
            if found_index > -1:
                # Just remap the ground-truth index to point the correct
                # predicted index.
                row_mappings[index] = found_index
        
        # Are there any indices of row_mappings that weren't matched?
        unmapped = len(pred_rows)
        if unmapped > 0:
            # Best option, unless anyone has any other ideas, is just to
            # match unmapped indices with predicted rows of smallest
            # pairwise [Euclidean] distance.

            # Unless there's only 1 unmapped pair, of course.
            if unmapped == 1:
                row_mappings[row_mappings < 0] = pred_rows[0]
            else:
                nm = (i for i, e in enumerate(row_mappings) if e < 0)
                for index in nm:
                    min_dist = -1
                    min_index = -1
                    min_index_index = -1
                    for i, match in enumerate(pred_rows):
                        dist = sla.norm(A_true[index] - A_pred[match])
                        if min_dist < 0 or min_dist > dist:
                            min_dist = dist
                            min_index = match
                            min_index_index = i
                    row_mappings[index] = match
                    pred_rows.pop(min_index_index)

        # All that... to do this.
        np.testing.assert_allclose(A_true[row_mappings], A_pred, rtol = rtol)

if __name__ == "__main__":
    unittest.main()
