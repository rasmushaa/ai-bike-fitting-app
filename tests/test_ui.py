import io

import cv2
import numpy as np
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ai_bike_fitting_app.analysis import AnalysisResult, FrameAngles

APP = Path(__file__).resolve().parents[1] / "app.py"


class UITests(unittest.TestCase):
    def test_initial_screen_needs_an_upload(self):
        app = AppTest.from_file(str(APP), default_timeout=15).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="analyze").disabled)
        self.assertEqual(app.radio[0].value, "Right")
        self.assertEqual(app.selectbox[0].value, 15)
        self.assertEqual(app.selectbox[0].options, ["15", "20", "25", "30"])

    def test_rerun_keeps_results_and_side_change_clears_them(self):
        upload = io.BytesIO(b"example upload")
        upload.name = "example.mp4"
        result = AnalysisResult([cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8))[1].tobytes()], [FrameAngles(0, 140, 150)], "right")
        with (
            patch("streamlit.file_uploader", return_value=upload),
            patch("ai_bike_fitting_app.ui.get_model_resource", return_value=(object(), threading.Lock())),
            patch("ai_bike_fitting_app.ui.process_video", return_value=result) as process,
        ):
            app = AppTest.from_file(str(APP), default_timeout=15).run()
            app.button(key="analyze").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(process.call_count, 1)
            self.assertIsNotNone(app.session_state["analysis_result"])
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(process.call_count, 1)
            self.assertIsNotNone(app.session_state["analysis_result"])
            with patch("ai_bike_fitting_app.ui.compile_video", return_value=b"mp4") as compile:
                self.assertIsNone(app.session_state["video_export"])
                app.button(key="compile").click().run()
                self.assertFalse(app.exception)
                self.assertEqual(compile.call_count, 1)
                self.assertEqual(process.call_count, 1)
                self.assertEqual(app.session_state["video_export"], b"mp4")
            app.selectbox[0].set_value(30).run()
            self.assertIsNone(app.session_state["analysis_result"])
            self.assertIsNone(app.session_state["video_export"])
            app.button(key="analyze").click().run()
            self.assertEqual(process.call_args.kwargs["fps"], 30)
            app.radio[0].set_value("Left").run()
            self.assertFalse(app.exception)
            self.assertEqual(process.call_count, 2)
            self.assertIsNone(app.session_state["analysis_result"])
            self.assertIsNone(app.session_state["video_export"])


if __name__ == "__main__":
    unittest.main()
