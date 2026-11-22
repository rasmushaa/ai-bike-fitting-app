import unittest

import numpy as np

from ai_bike_fitting_app.analysis import JOINTS, measure_angles, smooth_poses
from ai_bike_fitting_app.smoothing import smooth_positions
from test_angles import pose_points


class SmoothingTests(unittest.TestCase):
    def test_short_bounded_xy_gaps_are_interpolated(self):
        points = np.array([[np.nan, np.nan], [10, 20], [np.nan, np.nan], [30, 60],
                           [np.nan, np.nan], [np.nan, np.nan], [60, 120], [np.nan, np.nan]])
        result = smooth_positions(points)
        np.testing.assert_allclose(result[1:7], [[10, 20], [20, 40], [30, 60], [40, 80], [50, 100], [60, 120]], atol=1e-9)
        self.assertTrue(np.isnan(result[[0, -1]]).all())
        self.assertTrue(np.isnan(points[2]).all())

    def test_long_gaps_keep_filter_segments_separate(self):
        points = np.array([[20, 40]] * 30 + [[np.nan, np.nan]] * 3 + [[100, 200]] * 30)
        result = smooth_positions(points)
        np.testing.assert_allclose(result[:30], [[20, 40]] * 30, atol=1e-9)
        self.assertTrue(np.isnan(result[30:33]).all())
        np.testing.assert_allclose(result[33:], [[100, 200]] * 30, atol=1e-9)

    def test_empty_missing_and_short_segments(self):
        self.assertEqual(smooth_positions(np.empty((0, 2))).shape, (0, 2))
        self.assertTrue(np.isnan(smooth_positions(np.full((20, 2), np.nan))).all())
        points = np.array([[10, 20], [20, 30], [30, 40], [40, 50]])
        np.testing.assert_array_equal(smooth_positions(points), points)

    def test_reduces_xy_noise_and_preserves_quadratic_motion(self):
        time = np.arange(120) / 20
        motion = np.column_stack([100 + time * 20, 200 + time**2 * 10])
        noisy = motion + np.random.default_rng(7).normal(0, 4, motion.shape)
        result = smooth_positions(noisy)
        self.assertLess(np.mean((result - motion) ** 2), np.mean((noisy - motion) ** 2))
        np.testing.assert_allclose(smooth_positions(motion), motion, atol=1e-8)

    def test_confidence_and_selected_side_control_node_availability(self):
        poses = np.stack([pose_points() for _ in range(40)])
        poses[10:12, 13] = [999, 999, 0.1]
        poses[20:23, 13] = [999, 999, 0.1]
        result = smooth_poses(list(poses), "left")
        np.testing.assert_allclose(result[10:12, 13, :2], [[20, 50]] * 2, atol=1e-8)
        self.assertEqual(measure_angles(result[21], "left"), (None, 90))
        selected = {node for triplet in JOINTS["left"].values() for node in triplet}
        self.assertEqual(set(np.flatnonzero(result[0, :, 2])), selected)
        self.assertFalse(smooth_poses([None] * 10, "left").any())


if __name__ == "__main__":
    unittest.main()
