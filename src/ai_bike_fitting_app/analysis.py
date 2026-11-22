"""Bounded, batched video processing and 2D joint geometry."""

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from .smoothing import smooth_positions

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_SECONDS = 30
FPS = 15
FPS_OPTIONS = (15, 20, 25, 30)
INFERENCE_BATCH_SIZE = 8
KEYPOINT_CONFIDENCE = 0.5
OVERLAY_VERSION = 11

# RGB hex values shared by video overlays and angle distributions.
JOINT_COLORS = {"knee": "#2563EB", "elbow": "#DC2626"}

# COCO indices, in anatomical (rider's) left/right order.
JOINTS = {
    "left": {"knee": (11, 13, 15), "elbow": (5, 7, 9)},
    "right": {"knee": (12, 14, 16), "elbow": (6, 8, 10)},
}

class AnalysisError(RuntimeError):
    """An actionable video-processing failure."""


@dataclass(frozen=True)
class FrameAngles:
    time_s: float
    knee_deg: float | None
    elbow_deg: float | None


@dataclass(frozen=True)
class AnalysisResult:
    frames: list[bytes]
    samples: list[FrameAngles]
    side: str
    raw_samples: list[FrameAngles] | None = None
    fps: int = FPS

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.fps


def joint_angle(a, b, c) -> float | None:
    """Internal angle ABC in pixel coordinates; 180 degrees is straight."""
    points = np.asarray([a, b, c], dtype=float)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        return None
    u, v = points[0] - points[1], points[2] - points[1]
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator < 1e-8:
        return None
    cosine = np.clip(np.dot(u, v) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def valid_keypoints(keypoints: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(keypoints).all(axis=1)
        & (keypoints[:, 2] >= KEYPOINT_CONFIDENCE)
        & np.any(keypoints[:, :2] != 0, axis=1)
    )


def smooth_poses(poses: list[np.ndarray | None], side: str) -> np.ndarray:
    observations = np.full((len(poses), 17, 2), np.nan)
    for frame, pose in enumerate(poses):
        if pose is not None:
            valid = valid_keypoints(pose)
            observations[frame, valid] = pose[valid, :2]
    filtered = np.zeros((len(poses), 17, 3), dtype=float)
    for indices in JOINTS[side].values():
        for index in indices:
            xy = smooth_positions(observations[:, index])
            valid = np.isfinite(xy).all(axis=1)
            filtered[valid, index, :2] = xy[valid]
            # Availability of a cleaned coordinate, rather than YOLO confidence.
            filtered[valid, index, 2] = 1
    return filtered


def measure_angles(keypoints: np.ndarray, side: str) -> tuple[float | None, float | None]:
    valid = valid_keypoints(keypoints)
    angles = []
    for indices in JOINTS[side].values():
        angles.append(joint_angle(*keypoints[list(indices), :2]) if valid[list(indices)].all() else None)
    return tuple(angles)


def largest_pose(result) -> np.ndarray | None:
    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    points = result.keypoints.data.cpu().numpy()[int(np.argmax(area))]
    return points if points.shape == (17, 3) else None


def draw_overlay(frame, keypoints, side: str) -> None:
    if keypoints is not None:
        valid = valid_keypoints(keypoints)
        xy = np.zeros((len(keypoints), 2), dtype=int)
        xy[valid] = np.rint(keypoints[valid, :2]).astype(int)
        for joint, indices in JOINTS[side].items():
            # OpenCV takes BGR rather than RGB colors.
            color = tuple(bytes.fromhex(JOINT_COLORS[joint][1:]))[::-1]
            for a, b in zip(indices, indices[1:]):
                if valid[a] and valid[b]:
                    cv2.line(frame, tuple(xy[a]), tuple(xy[b]), color, 2, cv2.LINE_AA)
            for index in indices:
                if valid[index]:
                    cv2.circle(frame, tuple(xy[index]), 4, color, -1, cv2.LINE_AA)


def ffmpeg_run(arguments: list[str], error: str) -> None:
    try:
        executable = get_ffmpeg_exe()
        subprocess.run(
            [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *arguments],
            capture_output=True, check=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise AnalysisError(error) from exc
    except (RuntimeError, OSError) as exc:
        raise AnalysisError("Could not start the bundled video encoder. Reinstall the Python dependencies and retry.") from exc


def encode_frame(frame: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise AnalysisError("Could not create an analysis frame.")
    return encoded.tobytes()


def plot_limits(values) -> tuple[float, float, float, float]:
    """Central 90% cutoffs with padded, nonzero angle axes for the whole clip."""
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(finite):
        return 0.0, 180.0, 0.0, 180.0
    low, high = np.percentile(finite, [5, 95], method="nearest")
    padding = max((high - low) * 0.05, 5.0)
    lower = max(0.0, float(np.floor((low - padding) / 5) * 5))
    upper = min(180.0, float(np.ceil((high + padding) / 5) * 5))
    return float(low), float(high), lower, upper


def compose_frame(frame: np.ndarray, samples: list[FrameAngles], index: int, fps: int = FPS) -> np.ndarray:
    """Pose above two plot rows; traces and histograms include data up to index."""
    height, source_width = frame.shape[:2]
    width = max(640, source_width)
    row_height = 200
    canvas = np.full((height + row_height * 2, width, 3), 255, np.uint8)
    offset = (width - source_width) // 2
    canvas[:height, offset:offset + source_width] = frame
    duration = max(len(samples) / fps, 1 / fps)
    gray = (100, 100, 100)

    def text(value, x, y, color=gray, size=0.42):
        cv2.putText(canvas, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)

    for row, joint in enumerate(("knee", "elbow")):
        color = tuple(bytes.fromhex(JOINT_COLORS[joint][1:]))[::-1]
        top = height + row * row_height
        all_values = [getattr(s, f"{joint}_deg") for s in samples]
        _, _, lower, upper = plot_limits(all_values)
        valid_values = [v for v in all_values if v is not None]
        current = getattr(samples[index], f"{joint}_deg")
        label = f"{current:.1f} deg" if current is not None else "unavailable"
        text(f"{joint.title()} trace: {label}", 12, top + 22, color, 0.5)
        text(f"{joint.title()} distribution", width // 2 + 12, top + 22, color, 0.5)
        y0, y1 = top + 40, top + 158
        x0, x1 = 45, width // 2 - 20
        hx0, hx1 = width // 2 + 45, width - 20
        for value in (lower, (lower + upper) / 2, upper):
            y = round(y1 - (value - lower) / (upper - lower) * (y1 - y0))
            cv2.line(canvas, (x0, y), (x1, y), (235, 235, 235), 1)
            text(f"{value:g}", 9, y + 4)
        cv2.line(canvas, (x0, y0), (x0, y1), gray, 1)
        cv2.line(canvas, (x0, y1), (x1, y1), gray, 1)
        previous = None
        for sample in samples[:index + 1]:
            value = getattr(sample, f"{joint}_deg")
            if value is None:
                previous = None
                continue
            point = (round(x0 + sample.time_s / duration * (x1 - x0)), round(y1 - (value - lower) / (upper - lower) * (y1 - y0)))
            if previous is not None:
                visible, start, end = cv2.clipLine((x0, y0, x1 - x0 + 1, y1 - y0 + 1), previous, point)
                if visible:
                    cv2.line(canvas, start, end, color, 2, cv2.LINE_AA)
            previous = point
        cursor = round(x0 + samples[index].time_s / duration * (x1 - x0))
        cv2.line(canvas, (cursor, y0), (cursor, y1), (180, 180, 180), 1)
        if current is not None and lower <= current <= upper:
            cv2.circle(canvas, (cursor, round(y1 - (current - lower) / (upper - lower) * (y1 - y0))), 3, color, -1)
        text("0", x0 - 3, y1 + 16)
        text(f"{duration:.1f}", x1 - 22, y1 + 16)
        text("Time (s)", (x0 + x1) // 2 - 25, top + 192)

        values = [getattr(s, f"{joint}_deg") for s in samples[:index + 1]]
        values = [v for v in values if v is not None]
        bins = np.arange(0, 185, 5)
        counts, _ = np.histogram(values, bins=bins)
        full_counts, _ = np.histogram(valid_values, bins=bins)
        visible_bins = (bins[:-1] < upper) & (bins[1:] > lower)
        maximum = max(int(full_counts[visible_bins].max()), 1)
        for value in (0, maximum):
            y = round(y1 - value / maximum * (y1 - y0))
            cv2.line(canvas, (hx0, y), (hx1, y), (235, 235, 235), 1)
            text(value, hx0 - 32, y + 4)
        for bin_index, count in enumerate(counts):
            if count and visible_bins[bin_index]:
                left = max(hx0, round(hx0 + (bins[bin_index] - lower) / (upper - lower) * (hx1 - hx0)))
                right = min(hx1, round(hx0 + (bins[bin_index + 1] - lower) / (upper - lower) * (hx1 - hx0)))
                y = round(y1 - int(count) / maximum * (y1 - y0))
                cv2.rectangle(canvas, (left, y), (max(left, right - 1), y1), color, -1)
        cv2.line(canvas, (hx0, y0), (hx0, y1), gray, 1)
        cv2.line(canvas, (hx0, y1), (hx1, y1), gray, 1)
        for value in (lower, (lower + upper) / 2, upper):
            text(f"{value:g}", round(hx0 + (value - lower) / (upper - lower) * (hx1 - hx0)) - 10, y1 + 16)
        text("Angle (deg) / frame count", hx0 + 5, top + 192, size=0.38)
        if not values:
            text("No reliable measurements", hx0 + 5, y0 + 55, size=0.38)
    return canvas


def compile_video(result: AnalysisResult) -> bytes:
    """Encode the already-rendered frames only when the user requests an MP4."""
    if not result.frames:
        raise AnalysisError("There are no frames to compile.")
    with tempfile.TemporaryDirectory(prefix="bike-fit-export-") as directory:
        work = Path(directory)
        for index, frame in enumerate(result.frames):
            (work / f"frame-{index:04d}.jpg").write_bytes(frame)
        output = work / "result.mp4"
        ffmpeg_run([
            "-framerate", str(result.fps), "-i", str(work / "frame-%04d.jpg"), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-vf", "scale=in_range=pc:out_range=tv,pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p", "-color_range", "tv", "-movflags", "+faststart",
            "-threads", "2", str(output),
        ], "Could not compile the MP4. Please retry the download preparation.")
        return output.read_bytes()


def process_video(
    video: bytes,
    suffix: str,
    side: str,
    model,
    inference_lock,
    progress: Callable[[float], None] | None = None,
    fps: int = FPS,
) -> AnalysisResult:
    if fps not in FPS_OPTIONS:
        raise AnalysisError("Choose a processing FPS of 15, 20, 25, or 30.")
    if side not in JOINTS:
        raise AnalysisError("Choose the rider's left or right side.")
    if not video or len(video) > MAX_UPLOAD_BYTES:
        raise AnalysisError("Upload a non-empty video of at most 100 MB.")
    if suffix.lower() != ".mp4":
        raise AnalysisError("Upload an MP4 video.")

    with tempfile.TemporaryDirectory(prefix="bike-fit-") as directory:
        work = Path(directory)
        source = work / f"upload{suffix.lower()}"
        normalized = work / "input.avi"
        source.write_bytes(video)
        # FFmpeg applies rotation metadata. Uniform timestamps preserve duration,
        # including variable-frame-rate phone recordings. MJPEG is only temporary.
        ffmpeg_run([
            "-i", str(source), "-t", str(MAX_SECONDS), "-map", "0:v:0", "-an",
            "-vf", f"fps={fps},scale=w='min(960,iw)':h='min(960,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v", "mjpeg", "-q:v", "4", "-threads", "2", str(normalized),
        ], "Could not read this video. Try exporting it as an H.264 MP4 and uploading again.")

        capture = cv2.VideoCapture(str(normalized))
        frames = []
        raw_samples = []
        poses = []
        try:
            if not capture.isOpened():
                raise AnalysisError("Could not open the decoded video.")
            total = min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_SECONDS * fps)
            while len(poses) < MAX_SECONDS * fps:
                batch = []
                for _ in range(min(INFERENCE_BATCH_SIZE, MAX_SECONDS * fps - len(poses))):
                    ok, frame = capture.read()
                    if not ok:
                        break
                    batch.append(frame)
                if not batch:
                    break
                with inference_lock:
                    predictions = model.predict(batch, device="cpu", imgsz=640, conf=0.25, verbose=False)
                    # Keep only the small pose arrays for smoothing across batch boundaries.
                    detected_poses = [largest_pose(prediction) for prediction in predictions]
                for frame, detected in zip(batch, detected_poses):
                    raw_angles = measure_angles(detected, side) if detected is not None else (None, None)
                    raw_samples.append(FrameAngles(len(poses) / fps, *raw_angles))
                    poses.append(detected)
                    frames.append(encode_frame(frame))
                if progress is not None:
                    progress(0.8 * min(len(poses) / max(total, 1), 1.0))
        finally:
            capture.release()
        if not raw_samples:
            raise AnalysisError("This video contains no readable frames. Try another clip.")

        filtered_poses = smooth_poses(poses, side)
        samples = [FrameAngles(index / fps, *measure_angles(points, side)) for index, points in enumerate(filtered_poses)]
        composite_frames = []
        for index, encoded in enumerate(frames):
            frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            draw_overlay(frame, filtered_poses[index], side)
            composite_frames.append(encode_frame(compose_frame(frame, samples, index, fps)))
            if progress is not None and (index % 5 == 0 or index + 1 == len(frames)):
                progress(0.8 + 0.2 * (index + 1) / len(frames))
        return AnalysisResult(composite_frames, samples, side, raw_samples, fps)
