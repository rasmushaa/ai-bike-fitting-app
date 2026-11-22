import unittest
from types import SimpleNamespace

import numpy as np

from ai_bike_fitting_app.analysis import draw_overlay, joint_angle, largest_pose, measure_angles, plot_limits


class Tensor:
    """Small stand-in for CPU prediction tensors."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class Boxes:
    def __init__(self, values):
        self.xyxy = Tensor(values)

    def __len__(self):
        return len(self.xyxy.values)


def pose_points():
    points = np.full((17, 3), 0.9)
    # Left knee and elbow: 90 degrees. Right knee and elbow: 180 degrees.
    for indices, coordinates in [
        ((11, 13, 15), ((20, 20), (20, 50), (50, 50))),
        ((5, 7, 9), ((30, 20), (30, 40), (50, 40))),
        ((12, 14, 16), ((80, 20), (80, 50), (80, 80))),
        ((6, 8, 10), ((60, 20), (60, 40), (60, 60))),
    ]:
        points[list(indices), :2] = coordinates
    return points


class AngleTests(unittest.TestCase):
    def test_plot_limits_trim_tails_and_handle_constant_or_missing_data(self):
        low, high, lower, upper = plot_limits(range(100))
        self.assertEqual(sum(low <= value <= high for value in range(100)), 90)
        self.assertLessEqual(lower, low)
        self.assertGreaterEqual(upper, high)
        self.assertEqual(plot_limits([90] * 100), (90, 90, 85, 95))
        self.assertEqual(plot_limits([None, np.nan]), (0, 180, 0, 180))
        self.assertGreater(plot_limits([180])[3], plot_limits([180])[2])

    def test_missing_pose_does_not_add_text_to_video(self):
        frame = np.full((120, 160, 3), 255, np.uint8)
        draw_overlay(frame, None, "right")
        self.assertTrue((frame == 255).all())

    def test_internal_angles_in_pixels(self):
        self.assertAlmostEqual(joint_angle((0, 0), (0, 2), (2, 2)), 90)
        self.assertAlmostEqual(joint_angle((0, 0), (0, 2), (0, 4)), 180)
        self.assertAlmostEqual(joint_angle((0, 0), (2, 0), (0, 0)), 0)
        # Independent normalization of x/y would incorrectly make this 135°.
        self.assertAlmostEqual(joint_angle((0, 0), (4, 0), (0, 2)), 26.565051177)

    def test_missing_and_degenerate_coordinates(self):
        self.assertIsNone(joint_angle((1, 1), (1, 1), (2, 2)))
        self.assertIsNone(joint_angle((np.nan, 1), (1, 1), (2, 2)))

    def test_anatomical_sides(self):
        self.assertEqual(measure_angles(pose_points(), "left"), (90, 90))
        self.assertEqual(measure_angles(pose_points(), "right"), (180, 180))

    def test_confidence_filters_each_joint_independently(self):
        points = pose_points()
        points[13, 2] = 0.49
        self.assertEqual(measure_angles(points, "left"), (None, 90))
        points[13, 2] = 0.5
        self.assertEqual(measure_angles(points, "left"), (90, 90))
        points[5, :2] = (0, 0)
        self.assertEqual(measure_angles(points, "left"), (90, None))
        points[11, 0] = np.nan
        self.assertEqual(measure_angles(points, "left"), (None, None))

    def test_selects_largest_rider_and_handles_no_detection(self):
        poses = np.stack([pose_points(), pose_points() * 2])
        result = SimpleNamespace(
            boxes=Boxes([[0, 0, 10, 10], [0, 0, 100, 100]]),
            keypoints=SimpleNamespace(data=Tensor(poses)),
        )
        np.testing.assert_array_equal(largest_pose(result), poses[1])
        self.assertIsNone(largest_pose(SimpleNamespace(boxes=Boxes([]), keypoints=None)))


if __name__ == "__main__":
    unittest.main()
