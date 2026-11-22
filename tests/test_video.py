import os
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
from imageio_ffmpeg import get_ffmpeg_exe

from ai_bike_fitting_app.analysis import INFERENCE_BATCH_SIZE, AnalysisError, compile_video, ffmpeg_run, process_video
from test_angles import Boxes, Tensor, pose_points

FFMPEG = get_ffmpeg_exe()


class FakeModel:
    def __init__(self, detected=True):
        self.detected = detected
        self.calls = 0
        self.batch_sizes = []
        self.missing = set()

    def predict(self, frames, **kwargs):
        self.batch_sizes.append(len(frames))
        predictions = []
        for frame in frames:
            detected = self.detected and self.calls not in self.missing
            self.calls += 1
            predictions.append(SimpleNamespace(
                boxes=Boxes([[0, 0, 100, 100]]) if detected else Boxes([]),
                keypoints=SimpleNamespace(data=Tensor([pose_points()])) if detected else None,
            ))
        return predictions


class InputTests(unittest.TestCase):
    def test_invalid_upload_is_rejected_before_processing(self):
        for video, suffix, side in [(b"data", ".mov", "left"), (b"data", ".m4v", "left"), (b"", ".mp4", "left"), (b"data", ".txt", "left"), (b"data", ".mp4", "auto")]:
            with self.subTest(suffix=suffix, side=side), self.assertRaises(AnalysisError):
                process_video(video, suffix, side, FakeModel(), threading.Lock())


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)

    def make_video(self, seconds=1, fps=24):
        source = self.work / "source.mp4"
        subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", f"color=c=white:size=160x120:rate={fps}", "-t", str(seconds),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
        ], check=True, capture_output=True)
        return source

    def inspect_result(self, result):
        output = self.work / "output.mp4"
        output.write_bytes(compile_video(result))
        capture = cv2.VideoCapture(str(output))
        self.addCleanup(capture.release)
        self.assertTrue(capture.isOpened())
        self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), result.fps, places=2)
        ok, frame = capture.read()
        self.assertTrue(ok)
        self.assertTrue((frame < 100).any(), "The baked overlay should change the white input")
        probe = subprocess.run(
            [FFMPEG, "-hide_banner", "-i", str(output)], capture_output=True, text=True,
        )
        self.assertIn("Video: h264", probe.stderr)
        self.assertIn("yuv420p", probe.stderr)
        return capture

    def test_duration_angles_and_h264_overlay(self):
        source = self.make_video(seconds=2, fps=24)
        model = FakeModel()
        progress = []
        # A pip/uv install must work without any system FFmpeg on PATH.
        with patch.dict(os.environ, {"PATH": self.directory.name}), patch(
            "ai_bike_fitting_app.analysis.ffmpeg_run", wraps=ffmpeg_run,
        ) as media:
            result = process_video(source.read_bytes(), ".mp4", "left", model, threading.Lock(), progress.append)
        self.assertFalse(any("libx264" in call.args[0] for call in media.call_args_list))
        self.assertEqual(len(result.frames), 30)
        self.assertTrue(result.frames[0].startswith(b"\xff\xd8"))
        self.assertFalse(hasattr(result, "video"))
        capture = self.inspect_result(result)
        self.assertEqual(len(result.samples), 30)
        self.assertEqual(model.calls, 30)
        self.assertEqual(result.duration_s, 2)
        self.assertAlmostEqual(result.samples[0].knee_deg, 90)
        self.assertEqual(result.raw_samples[0].knee_deg, 90)
        self.assertAlmostEqual(result.samples[-1].elbow_deg, 90)
        self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT) / result.fps, 2)
        self.assertEqual(progress[-1], 1)

    def test_clips_to_thirty_seconds_and_keeps_missing_values(self):
        source = self.make_video(seconds=31, fps=5)
        result = process_video(source.read_bytes(), ".mp4", "right", FakeModel(False), threading.Lock())
        self.assertEqual(result.duration_s, 30)
        self.assertEqual(len(result.samples), 450)
        self.assertTrue(all(sample.knee_deg is None and sample.elbow_deg is None for sample in result.samples))
        capture = self.inspect_result(result)
        self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), 450)

    def test_selected_fps_controls_processing_timestamps_and_export(self):
        source = self.make_video()
        for fps in (15, 20, 25, 30):
            with self.subTest(fps=fps):
                model = FakeModel()
                result = process_video(source.read_bytes(), ".mp4", "left", model, threading.Lock(), fps=fps)
                self.assertEqual(result.fps, fps)
                self.assertEqual(model.calls, fps)
                self.assertEqual(model.batch_sizes, [INFERENCE_BATCH_SIZE] * (fps // INFERENCE_BATCH_SIZE)
                                 + ([fps % INFERENCE_BATCH_SIZE] if fps % INFERENCE_BATCH_SIZE else []))
                self.assertEqual(len(result.samples), fps)
                self.assertEqual(result.duration_s, 1)
                self.assertAlmostEqual(result.samples[-1].time_s, (fps - 1) / fps)
                self.assertAlmostEqual(result.raw_samples[-1].time_s, (fps - 1) / fps)
                capture = self.inspect_result(result)
                self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_COUNT), fps)

    def test_rotation_metadata_is_applied(self):
        source = self.make_video()
        rotated = self.work / "rotated.mp4"
        options = subprocess.run(
            [FFMPEG, "-hide_banner", "-h", "full"], capture_output=True, text=True, check=True,
        ).stdout
        rotation_input = ["-display_rotation", "90"] if "-display_rotation" in options else []
        rotation_output = [] if rotation_input else ["-metadata:s:v:0", "rotate=90"]
        subprocess.run([
            FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *rotation_input,
            "-i", str(source), "-c", "copy", *rotation_output, str(rotated),
        ], check=True, capture_output=True)
        result = process_video(rotated.read_bytes(), ".mp4", "left", FakeModel(), threading.Lock())
        capture = self.inspect_result(result)
        self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 640)
        self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 560)

    def test_corrupt_upload_has_actionable_error(self):
        with self.assertRaisesRegex(AnalysisError, "Could not read this video"):
            process_video(b"invalid mp4", ".mp4", "left", FakeModel(), threading.Lock())

    def test_short_gaps_are_filled_but_long_gaps_stay_missing(self):
        source = self.make_video(seconds=1, fps=30)
        for missing in ({3, 4}, {3, 4, 5, 6}, {7, 8}, {7, 8, 9}):
            with self.subTest(missing=missing):
                model = FakeModel()
                model.missing = missing
                result = process_video(source.read_bytes(), ".mp4", "left", model, threading.Lock())
                self.assertEqual(len(result.samples), 15)
                for index in missing:
                    self.assertIsNone(result.raw_samples[index].knee_deg)
                    if len(missing) < 3:
                        self.assertAlmostEqual(result.samples[index].knee_deg, 90)
                    else:
                        self.assertIsNone(result.samples[index].knee_deg)
                for sample in result.samples:
                    if sample.knee_deg is not None:
                        self.assertAlmostEqual(sample.knee_deg, 90)


if __name__ == "__main__":
    unittest.main()
