import unittest

import numpy as np
from nv_ground_state import bvec_rotation
from nv_ground_state import get_bfields


class NVgroundTest(unittest.TestCase):
    def test_get_tetrahedral_fields(self):
        """
        Tests whether or not method to rotate B-field (having the effect of switching NV-configuration) works.
        :return:
        """

        # z(0) = [0,0,1]
        # z(1) = [0, (8/9)**0.5, -1/3]
        # z(2) = [-(6/9)**0.5, -(8/36)**0.5, -1/3]
        # z(3) = [(6/9)**0.5, -(8/36)**0.5, -1/3]

        # Test 1: set unitized B-field parallel to z(0) and new z-axis to z(1)
        #self.assertEqual([0, -(8/9)**0.5, -1/3], bvec_rotation([0, 0, 1], [0, (8/9)**0.5, -1/3]))

        # Test 2: set unitized B-field parallel to y(0) and new z-axis to z(2)
        newB = bvec_rotation([0, 1, 0], [-(6/9)**0.5, -(8/36)**0.5, -1/3])
        ang = np.arccos(newB[2])
        self.assertEqual(2.06, round(ang, 2))

        # Test 3 : set B-field to (4.65, 1.45, 3.57) in the z(0) basis and new z-axis to z(3)
        newB = bvec_rotation([2, -3, 7], [(6/9)**0.5, -(8/36)**0.5, -1/3])
        ang = np.arccos(newB[2]/np.linalg.norm(newB))
        self.assertEqual(1.48, round(ang, 2))
        #self.assertEqual([3.67, -3.38, 3.40], bvec_rotation([4.65, 1.45, 3.57], [(6/9)**0.5, -(8/36)**0.5, -1/3]))

        print(get_bfields([0,0,1], 0))


if __name__ == '__main__':
    unittest.main()
