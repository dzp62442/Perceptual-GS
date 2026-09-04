"""Materialize an OmniScene bin as a Perceptual-GS scene cache."""

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image
from plyfile import PlyData

from scene.dataset_readers import storePly


PREPARED_FORMAT_VERSION = 2
COORDINATE_CONVENTION_VERSION = "omniscene_opencv_c2w_to_blender_yz_v1"
GAMMA = 1.5
ENHANCEMENT_THRESHOLD = 0.05
SMOOTHING_THRESHOLD = 0.3
POOLING_KERNEL_SIZE = 5
TRAIN_VIEW_COUNT = 6
TARGET_VIEW_COUNT = 18
FLIP_YZ = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def _atomic_write_json(path: Path, payload: Dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    os.replace(str(temporary_path), str(path))


def _tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected RGB tensor [3,H,W], got {}".format(tuple(image.shape)))
    image = image.detach().cpu().clamp(0.0, 1.0)
    return (
        (image * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()
    )


def extract_sensitivity_map(image: torch.Tensor) -> torch.Tensor:
    """CPU equivalent of the repository's original ``preprocess.py`` path."""

    rgb = _tensor_to_uint8(image)
    grayscale = Image.fromarray(rgb, mode="RGB").convert("L")
    grayscale = grayscale.point(
        lambda value: 255.0 * (float(value) / 255.0) ** (1.0 / GAMMA)
    )
    gray = torch.from_numpy(np.asarray(grayscale, dtype=np.float32) / 255.0)[None, None]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )[None, None]
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    )[None, None]
    edge_x = functional.conv2d(gray, sobel_x, padding=1)
    edge_y = functional.conv2d(gray, sobel_y, padding=1)
    edge = torch.sqrt(edge_x.square() + edge_y.square())
    enhanced = (edge > ENHANCEMENT_THRESHOLD).float()
    pooled = functional.avg_pool2d(enhanced, kernel_size=POOLING_KERNEL_SIZE)
    smoothed = functional.interpolate(pooled, size=edge.shape[-2:], mode="nearest")
    return (smoothed > SMOOTHING_THRESHOLD).float()[0, 0]


def _save_rgb_and_sensitivity(image: torch.Tensor, image_path: Path, sensitivity_path: Path) -> None:
    Image.fromarray(_tensor_to_uint8(image), mode="RGB").save(image_path)
    sensitivity = extract_sensitivity_map(image)
    sensitivity_array = (sensitivity * 255.0).clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(sensitivity_array, mode="L").save(sensitivity_path)


def opencv_c2w_to_blender(c2w_cv: np.ndarray) -> np.ndarray:
    c2w_cv = np.asarray(c2w_cv, dtype=np.float32)
    if c2w_cv.shape != (4, 4) or not np.isfinite(c2w_cv).all():
        raise ValueError("Invalid OpenCV c2w matrix")
    return c2w_cv @ FLIP_YZ


def _frame_record(
    file_path: str,
    c2w_cv: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: Tuple[int, int],
) -> Dict:
    height, width = resolution
    c2w_gl = opencv_c2w_to_blender(c2w_cv.detach().cpu().numpy())
    intrinsic = intrinsics.detach().cpu().numpy()
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("Invalid camera intrinsic matrix")
    return {
        "file_path": file_path,
        "transform_matrix": c2w_gl.tolist(),
        "fl_x": float(intrinsic[0, 0]),
        "fl_y": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "w": int(width),
        "h": int(height),
    }


def _image_file_valid(path: Path, resolution: Tuple[int, int], mode: Optional[str] = None) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            if image.size != (resolution[1], resolution[0]):
                return False
            if mode is not None and image.mode != mode:
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def _read_transform_frames(path: Path, expected_count: int) -> Optional[List[Dict]]:
    try:
        with path.open("r", encoding="utf-8") as transforms_file:
            payload = json.load(transforms_file)
        if payload.get("coordinate_convention") != COORDINATE_CONVENTION_VERSION:
            return None
        frames = payload.get("frames")
        if not isinstance(frames, list) or len(frames) != expected_count:
            return None
        for frame in frames:
            if not isinstance(frame, dict):
                return None
            if any(key not in frame for key in (
                "file_path", "transform_matrix", "fl_x", "fl_y", "cx", "cy", "w", "h"
            )):
                return None
            transform = np.asarray(frame["transform_matrix"], dtype=np.float64)
            intrinsics = np.asarray(
                [frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]],
                dtype=np.float64,
            )
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                return None
            if not np.isfinite(intrinsics).all() or np.any(intrinsics[:2] <= 0):
                return None
        return frames
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def prepared_scene_complete(
    scene_dir: os.PathLike,
    bin_token: str,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    mode: Optional[str] = None,
    data_root: Optional[os.PathLike] = None,
    scene_index: Optional[int] = None,
) -> bool:
    """Validate the durable prepared-cache contract without loading raw data."""

    scene_dir = Path(scene_dir)
    resolution = tuple(int(value) for value in resolution)
    required = (
        scene_dir / "transforms_train.json",
        scene_dir / "transforms_test.json",
        scene_dir / "points3d.ply",
        scene_dir / "meta.json",
    )
    if not scene_dir.is_dir() or any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        return False

    try:
        with (scene_dir / "meta.json").open("r", encoding="utf-8") as meta_file:
            meta = json.load(meta_file)
        if meta.get("format_version") != PREPARED_FORMAT_VERSION:
            return False
        if meta.get("coordinate_convention") != COORDINATE_CONVENTION_VERSION:
            return False
        if meta.get("bin_token") != bin_token:
            return False
        if meta.get("resolution") != list(resolution):
            return False
        if int(meta.get("n_views", -1)) != TRAIN_VIEW_COUNT:
            return False
        if int(meta.get("target_views", -1)) != TARGET_VIEW_COUNT:
            return False
        if not math.isclose(
            float(meta.get("confidence_threshold")), float(confidence_threshold),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            return False
        if int(meta.get("initial_point_count", 0)) <= 0:
            return False
        if meta.get("sensitivity") != {
            "gamma": GAMMA,
            "enhancement_threshold": ENHANCEMENT_THRESHOLD,
            "smoothing_threshold": SMOOTHING_THRESHOLD,
            "pooling_kernel_size": POOLING_KERNEL_SIZE,
        }:
            return False
        if mode is not None and meta.get("split") != mode:
            return False
        if scene_index is not None and int(meta.get("scene_index", -1)) != int(scene_index):
            return False
        if data_root is not None:
            expected_root = os.path.realpath(os.path.expanduser(str(data_root)))
            if meta.get("data_root") != expected_root:
                return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False

    try:
        ply_data = PlyData.read(str(scene_dir / "points3d.ply"))
        vertex = ply_data["vertex"]
        if len(vertex) != int(meta["initial_point_count"]):
            return False
        if not all(name in vertex.data.dtype.names for name in (
            "x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"
        )):
            return False
        points = np.stack((vertex["x"], vertex["y"], vertex["z"]), axis=1)
        if not np.isfinite(points).all():
            return False
    except (OSError, KeyError, TypeError, ValueError):
        return False

    train_frames = _read_transform_frames(scene_dir / "transforms_train.json", TRAIN_VIEW_COUNT)
    test_frames = _read_transform_frames(scene_dir / "transforms_test.json", TARGET_VIEW_COUNT)
    if train_frames is None or test_frames is None:
        return False
    if any(
        int(frame["w"]) != resolution[1] or int(frame["h"]) != resolution[0]
        for frame in train_frames + test_frames
    ):
        return False

    expected = {
        "train": ["{:03d}.png".format(index) for index in range(TRAIN_VIEW_COUNT)],
        "test": ["{:03d}.png".format(index) for index in range(TARGET_VIEW_COUNT)],
    }
    if meta.get("train_images") != expected["train"] or meta.get("test_images") != expected["test"]:
        return False

    for split, frames in (("train", train_frames), ("test", test_frames)):
        image_dir = scene_dir / split
        sensitivity_dir = image_dir / "sensitivity_maps"
        if not image_dir.is_dir() or not sensitivity_dir.is_dir():
            return False
        actual_images = sorted(path.name for path in image_dir.glob("*.png"))
        actual_sensitivity = sorted(path.name for path in sensitivity_dir.glob("*.png"))
        if actual_images != expected[split] or actual_sensitivity != expected[split]:
            return False
        expected_frame_paths = ["{}/{}".format(split, Path(name).stem) for name in expected[split]]
        if [frame.get("file_path") for frame in frames] != expected_frame_paths:
            return False
        for name in expected[split]:
            if not _image_file_valid(image_dir / name, resolution, mode="RGB"):
                return False
            if not _image_file_valid(sensitivity_dir / name, resolution, mode="L"):
                return False
    return True


def build_initial_point_cloud(
    images: torch.Tensor,
    depths: torch.Tensor,
    confidences: torch.Tensor,
    intrinsics: torch.Tensor,
    c2ws_cv: torch.Tensor,
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backproject six metric-depth maps into the bin's LiDAR world frame."""

    if depths is None or confidences is None:
        raise ValueError("Metric depth and confidence are required for OmniScene initialization")
    if images.shape[0] != TRAIN_VIEW_COUNT:
        raise ValueError("OmniScene point cloud requires exactly six context views")

    point_chunks = []
    color_chunks = []
    for image, depth, confidence, intrinsic, c2w_cv in zip(
        images, depths, confidences, intrinsics, c2ws_cv
    ):
        image_np = _tensor_to_uint8(image)
        depth_np = np.asarray(depth.detach().cpu().numpy(), dtype=np.float32)
        confidence_np = np.asarray(confidence.detach().cpu().numpy(), dtype=np.float32)
        intrinsic_np = np.asarray(intrinsic.detach().cpu().numpy(), dtype=np.float32)
        c2w_np = np.asarray(c2w_cv.detach().cpu().numpy(), dtype=np.float32)
        valid = (
            np.isfinite(depth_np)
            & (depth_np > 0.0)
            & np.isfinite(confidence_np)
            & (confidence_np > float(confidence_threshold))
        )
        if not np.any(valid):
            continue

        height, width = depth_np.shape
        pixel_x, pixel_y = np.meshgrid(
            np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
        )
        fx = intrinsic_np[0, 0]
        fy = intrinsic_np[1, 1]
        cx = intrinsic_np[0, 2]
        cy = intrinsic_np[1, 2]
        z = depth_np
        x = (pixel_x - cx) * z / fx
        y = (pixel_y - cy) * z / fy
        camera_points = np.stack((x, y, z, np.ones_like(z)), axis=-1)[valid]
        world_points = (c2w_np @ camera_points.T).T[:, :3]
        colors = image_np[valid]
        finite = np.isfinite(world_points).all(axis=1)
        if np.any(finite):
            point_chunks.append(world_points[finite])
            color_chunks.append(colors[finite])

    if not point_chunks:
        raise ValueError(
            "No valid OmniScene depth points remain after confidence > {} filtering".format(
                confidence_threshold
            )
        )
    points = np.concatenate(point_chunks, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(color_chunks, axis=0).astype(np.uint8, copy=False)
    return points, colors


def _write_scene_contents(
    scene_data: Dict,
    scene_dir: Path,
    scene_index: int,
    mode: str,
    data_root: os.PathLike,
    confidence_threshold: float,
) -> None:
    context = scene_data["context"]
    target = scene_data["target"]
    context_images = context["image"]
    target_images = target["image"]
    if context_images.shape[0] != TRAIN_VIEW_COUNT or target_images.shape[0] != TARGET_VIEW_COUNT:
        raise ValueError("OmniScene requires 6 context and 18 target images")
    resolution = (int(context_images.shape[2]), int(context_images.shape[3]))
    if tuple(target_images.shape[2:]) != resolution:
        raise ValueError("Context and target resolutions do not match")

    train_dir = scene_dir / "train"
    test_dir = scene_dir / "test"
    train_sensitivity_dir = train_dir / "sensitivity_maps"
    test_sensitivity_dir = test_dir / "sensitivity_maps"
    for directory in (train_dir, test_dir, train_sensitivity_dir, test_sensitivity_dir):
        directory.mkdir(parents=True, exist_ok=True)

    train_names = ["{:03d}.png".format(index) for index in range(TRAIN_VIEW_COUNT)]
    test_names = ["{:03d}.png".format(index) for index in range(TARGET_VIEW_COUNT)]
    for image, name in zip(context_images, train_names):
        _save_rgb_and_sensitivity(image, train_dir / name, train_sensitivity_dir / name)
    for image, name in zip(target_images, test_names):
        _save_rgb_and_sensitivity(image, test_dir / name, test_sensitivity_dir / name)

    train_frames = [
        _frame_record(
            "train/{}".format(Path(name).stem), c2w, intrinsic, resolution
        )
        for name, c2w, intrinsic in zip(
            train_names, context["extrinsics"], context["intrinsics"]
        )
    ]
    test_frames = [
        _frame_record(
            "test/{}".format(Path(name).stem), c2w, intrinsic, resolution
        )
        for name, c2w, intrinsic in zip(
            test_names, target["extrinsics"], target["intrinsics"]
        )
    ]
    _atomic_write_json(
        scene_dir / "transforms_train.json",
        {"coordinate_convention": COORDINATE_CONVENTION_VERSION, "frames": train_frames},
    )
    _atomic_write_json(
        scene_dir / "transforms_test.json",
        {"coordinate_convention": COORDINATE_CONVENTION_VERSION, "frames": test_frames},
    )

    points, colors = build_initial_point_cloud(
        context_images,
        context["depth_m"],
        context["confidence"],
        context["intrinsics"],
        context["extrinsics"],
        confidence_threshold,
    )
    storePly(str(scene_dir / "points3d.ply"), points, colors)

    _atomic_write_json(
        scene_dir / "meta.json",
        {
            "format_version": PREPARED_FORMAT_VERSION,
            "coordinate_convention": COORDINATE_CONVENTION_VERSION,
            "bin_token": scene_data["bin_token"],
            "scene_index": int(scene_index),
            "split": mode,
            "resolution": list(resolution),
            "n_views": TRAIN_VIEW_COUNT,
            "target_views": TARGET_VIEW_COUNT,
            "confidence_threshold": float(confidence_threshold),
            "sensitivity": {
                "gamma": GAMMA,
                "enhancement_threshold": ENHANCEMENT_THRESHOLD,
                "smoothing_threshold": SMOOTHING_THRESHOLD,
                "pooling_kernel_size": POOLING_KERNEL_SIZE,
            },
            "train_images": train_names,
            "test_images": test_names,
            "initial_point_count": int(points.shape[0]),
            "data_root": os.path.realpath(os.path.expanduser(str(data_root))),
        },
    )


def _remove_cache_directory(path: Path, root: Path) -> None:
    root_resolved = root.resolve()
    path_resolved = path.resolve()
    if path_resolved.parent != root_resolved or path_resolved == root_resolved:
        raise RuntimeError("Refusing to remove cache directory outside prepared root: {}".format(path))
    if path.is_symlink():
        raise RuntimeError("Refusing to replace a symlinked prepared scene: {}".format(path))
    shutil.rmtree(str(path))


def preprocess_scene(
    scene_data: Dict,
    output_root: os.PathLike,
    scene_name: str,
    scene_index: int,
    mode: str,
    data_root: os.PathLike,
    confidence_threshold: float = 0.3,
) -> Path:
    """Create or reuse a complete cache, publishing new contents atomically."""

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene_dir = output_root / scene_name
    resolution = (
        int(scene_data["context"]["image"].shape[2]),
        int(scene_data["context"]["image"].shape[3]),
    )
    if prepared_scene_complete(
        scene_dir,
        scene_data["bin_token"],
        resolution,
        confidence_threshold,
        mode=mode,
        data_root=data_root,
        scene_index=scene_index,
    ):
        return scene_dir

    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".{}.tmp-".format(scene_name), dir=str(output_root))
    )
    try:
        _write_scene_contents(
            scene_data,
            temporary_dir,
            scene_index,
            mode,
            data_root,
            confidence_threshold,
        )
        if not prepared_scene_complete(
            temporary_dir,
            scene_data["bin_token"],
            resolution,
            confidence_threshold,
            mode=mode,
            data_root=data_root,
            scene_index=scene_index,
        ):
            raise RuntimeError("New OmniScene prepared cache failed validation: {}".format(scene_name))
        if scene_dir.exists():
            _remove_cache_directory(scene_dir, output_root)
        os.replace(str(temporary_dir), str(scene_dir))
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(str(temporary_dir), ignore_errors=True)
        raise
    return scene_dir
