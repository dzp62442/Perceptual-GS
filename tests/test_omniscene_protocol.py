import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from comp_svfgs.dataset_omniscene import (
    CAMERA_TYPES,
    CENTER150_SAMPLE_COUNT,
    OmniSceneDataset,
    load_center150_tokens,
    resolve_condition_paths,
)
from comp_svfgs.omniscene_preprocess import (
    FLIP_YZ,
    build_initial_point_cloud,
    prepared_scene_complete,
    preprocess_scene,
)
from scene.dataset_readers import readCamerasFromTransforms
from scripts import run_omniscene
from utils.graphics_utils import getProjectionMatrixFromIntrinsics


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _synthetic_sample(height=8, width=10, token="scene00000000000000000000000000000001_bin001"):
    context_images = torch.zeros((6, 3, height, width), dtype=torch.float32)
    target_images = torch.zeros((18, 3, height, width), dtype=torch.float32)
    for index in range(6):
        context_images[index] = (index + 1) / 10.0
    for index in range(18):
        target_images[index] = (index + 1) / 20.0
    intrinsics = torch.tensor(
        [[8.0, 0.0, width / 2.0], [0.0, 9.0, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    context_intrinsics = intrinsics[None].repeat(6, 1, 1)
    target_intrinsics = intrinsics[None].repeat(18, 1, 1)
    context_poses = torch.eye(4, dtype=torch.float32)[None].repeat(6, 1, 1)
    target_poses = torch.eye(4, dtype=torch.float32)[None].repeat(18, 1, 1)
    for index in range(6):
        context_poses[index, 0, 3] = float(index)
    depths = torch.ones((6, height, width), dtype=torch.float32) * 2.0
    confidences = torch.ones((6, height, width), dtype=torch.float32) * 0.5
    return {
        "bin_token": token,
        "scene": token,
        "context": {
            "image": context_images,
            "intrinsics": context_intrinsics,
            "extrinsics": context_poses,
            "depth_m": depths,
            "confidence": confidences,
        },
        "target": {
            "image": target_images,
            "intrinsics": target_intrinsics,
            "extrinsics": target_poses,
        },
    }


class Center150ManifestTest(unittest.TestCase):
    def test_center150_is_read_only_strict_and_ordered(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            version_root = root / "interp_12Hz_trainval"
            info_root = version_root / "bin_infos_3.2m"
            info_root.mkdir(parents=True)
            tokens = []
            for index in range(CENTER150_SAMPLE_COUNT):
                token = "scene{:032x}_bin{:03d}".format(index + 1, index)
                tokens.append(token)
                (info_root / "{}.pkl".format(token)).write_bytes(b"info")
            _write_json(version_root / "bins_center150_v1.json", {"bins": tokens})
            self.assertEqual(load_center150_tokens(root), tokens)

            _write_json(version_root / "bins_center150_v1.json", {"bins": tokens[:-1]})
            with self.assertRaisesRegex(ValueError, "150 unique bins"):
                load_center150_tokens(root)


class DatasetLoadingTest(unittest.TestCase):
    def _create_view_files(self, root, camera, frame_index, color):
        stem = "{}_{}".format(camera, frame_index)
        source = root / "samples" / camera / "{}.jpg".format(stem)
        image_path = root / "samples_small" / camera / "{}.jpg".format(stem)
        parameter_path = root / "samples_param_small" / camera / "{}.json".format(stem)
        depth_path = root / "samples_dptm_small" / camera / "{}_dpt.npy".format(stem)
        confidence_path = root / "samples_dptm_small" / camera / "{}_conf.npy".format(stem)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        parameter_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((224, 400, 3), color, dtype=np.uint8), mode="RGB").save(image_path)
        _write_json(
            parameter_path,
            {"camera_intrinsic": [[320.0, 0.0, 200.0], [0.0, 300.0, 112.0], [0.0, 0.0, 1.0]]},
        )
        np.save(depth_path, np.full((224, 400), frame_index + 1.0, dtype=np.float32))
        np.save(confidence_path, np.full((224, 400), 0.5, dtype=np.float32))
        return "/datasets/nuScenes/{}".format(source.relative_to(root).as_posix())

    def test_context_target_order_resize_and_metric_depth(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            version_root = root / "interp_12Hz_trainval"
            info_root = version_root / "bin_infos_3.2m"
            info_root.mkdir(parents=True)
            token = "scene00000000000000000000000000000001_bin001"
            _write_json(version_root / "bins_train_3.2m.json", {"bins": [token]})
            sensor_info = {"LIDAR_TOP": [{}, {}, {}]}
            for camera_index, camera in enumerate(CAMERA_TYPES):
                records = []
                for frame_index in range(3):
                    color = 10 + camera_index * 30 + frame_index * 5
                    c2w = np.eye(4, dtype=np.float32)
                    c2w[0, 3] = camera_index
                    c2w[1, 3] = frame_index
                    records.append(
                        {
                            "data_path": self._create_view_files(
                                root, camera, frame_index, color
                            ),
                            "sensor2lidar_transform": c2w,
                        }
                    )
                sensor_info[camera] = records
            with (info_root / "{}.pkl".format(token)).open("wb") as info_file:
                pickle.dump({"sensor_info": sensor_info}, info_file)

            dataset = OmniSceneDataset(root, mode="train", resolution=(112, 200))
            sample = dataset[0]
            self.assertEqual(tuple(sample["context"]["image"].shape), (6, 3, 112, 200))
            self.assertEqual(tuple(sample["target"]["image"].shape), (18, 3, 112, 200))
            self.assertEqual(tuple(sample["context"]["depth_m"].shape), (6, 112, 200))
            self.assertAlmostEqual(sample["context"]["intrinsics"][0, 0, 0].item(), 160.0)
            self.assertAlmostEqual(sample["context"]["intrinsics"][0, 1, 1].item(), 150.0)
            self.assertTrue(torch.equal(sample["target"]["image"][12:], sample["context"]["image"]))
            self.assertAlmostEqual(
                sample["target"]["image"][0, 0, 0, 0].item(), 15.0 / 255.0,
                places=5,
            )
            self.assertAlmostEqual(
                sample["target"]["image"][1, 0, 0, 0].item(), 20.0 / 255.0,
                places=5,
            )

    def test_path_mapping_covers_samples_and_sweeps(self):
        samples = resolve_condition_paths("/tmp/data/samples/CAM_FRONT/a.jpg")
        sweeps = resolve_condition_paths("/tmp/data/sweeps/CAM_FRONT/b.jpg")
        self.assertIn("samples_param_small", str(samples["parameter"]))
        self.assertIn("samples_dptm_small", str(samples["depth"]))
        self.assertIn("sweeps_small", str(sweeps["image"]))
        self.assertIn("sweeps_dptm_small", str(sweeps["confidence"]))


class PreprocessAndCameraTest(unittest.TestCase):
    def test_point_cloud_filter_is_strict_and_geometry_is_opencv(self):
        sample = _synthetic_sample(height=5, width=5)
        sample["context"]["confidence"].fill_(0.3)
        sample["context"]["confidence"][0, 2, 2] = 0.31
        sample["context"]["depth_m"][0, 2, 2] = 3.0
        points, colors = build_initial_point_cloud(
            sample["context"]["image"],
            sample["context"]["depth_m"],
            sample["context"]["confidence"],
            sample["context"]["intrinsics"],
            sample["context"]["extrinsics"],
            0.3,
        )
        self.assertEqual(points.shape, (1, 3))
        expected = np.array(
            [
                (2.0 - 2.5) / 8.0 * 3.0,
                (2.0 - 2.5) / 9.0 * 3.0,
                3.0,
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(points[0], expected, atol=1e-6)
        self.assertEqual(colors.shape, (1, 3))

    def test_prepared_cache_intrinsics_pose_and_rgb_alpha(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sample = _synthetic_sample()
            scene_name = "001_{}".format(sample["bin_token"])
            scene_dir = preprocess_scene(
                sample, root / "prepared", scene_name, 1, "center150", root, 0.3
            )
            self.assertTrue(
                prepared_scene_complete(
                    scene_dir, sample["bin_token"], (8, 10), 0.3,
                    mode="center150", data_root=root, scene_index=1,
                )
            )
            train_payload = json.loads((scene_dir / "transforms_train.json").read_text())
            frame = train_payload["frames"][0]
            c2w_gl = np.asarray(frame["transform_matrix"], dtype=np.float32)
            recovered_c2w_cv = c2w_gl.copy()
            recovered_c2w_cv[:3, 1:3] *= -1
            np.testing.assert_allclose(recovered_c2w_cv, np.eye(4), atol=1e-6)
            np.testing.assert_allclose(c2w_gl, np.eye(4) @ FLIP_YZ, atol=1e-6)

            camera_info = readCamerasFromTransforms(
                str(scene_dir), "transforms_train.json", False, "sensitivity_maps"
            )[0]
            self.assertIsNone(camera_info.alpha)
            self.assertAlmostEqual(camera_info.fx, 8.0)
            self.assertAlmostEqual(camera_info.fy, 9.0)
            self.assertAlmostEqual(camera_info.cx, 5.0)
            self.assertAlmostEqual(camera_info.cy, 4.0)
            np.testing.assert_allclose(camera_info.R, np.eye(3), atol=1e-6)
            np.testing.assert_allclose(camera_info.T, np.zeros(3), atol=1e-6)

    def test_projection_keeps_off_center_principal_point(self):
        projection = getProjectionMatrixFromIntrinsics(
            0.01, 100.0, fx=100.0, fy=120.0, cx=70.0, cy=40.0,
            width=200, height=100,
        )
        self.assertAlmostEqual(projection[0, 2].item(), -0.3, places=6)
        self.assertAlmostEqual(projection[1, 2].item(), -0.2, places=6)
        point = torch.tensor([0.0, 0.0, 2.0, 1.0])
        projected = projection @ point
        self.assertAlmostEqual((projected[0] / projected[3]).item(), -0.3, places=6)
        self.assertAlmostEqual((projected[1] / projected[3]).item(), -0.2, places=6)

    def test_original_blender_fov_and_real_alpha_remain_supported(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "train" / "sensitivity_maps").mkdir(parents=True)
            Image.new("RGBA", (10, 8), color=(100, 120, 140, 128)).save(
                root / "train" / "000.png"
            )
            Image.new("L", (10, 8), color=255).save(
                root / "train" / "sensitivity_maps" / "000.png"
            )
            _write_json(
                root / "transforms_train.json",
                {
                    "camera_angle_x": 0.8,
                    "frames": [
                        {"file_path": "train/000", "transform_matrix": np.eye(4).tolist()}
                    ],
                },
            )
            camera_info = readCamerasFromTransforms(
                str(root), "transforms_train.json", False, "sensitivity_maps"
            )[0]
            self.assertAlmostEqual(camera_info.FovX, 0.8)
            self.assertIsNone(camera_info.fx)
            self.assertIsNotNone(camera_info.alpha)
            self.assertEqual(np.asarray(camera_info.alpha)[0, 0, 0], 128)


class RunnerProtocolTest(unittest.TestCase):
    EVAL_ITERATIONS = (1, 2, 3)

    def _make_outputs(self, model_path, prepared_scene, training_times=(1.0, 2.0, 3.0)):
        model_path.mkdir(parents=True, exist_ok=True)
        for name in ("cfg_args", "input.ply", "cameras.json"):
            (model_path / name).write_bytes(b"x")
        final_dir = model_path / "point_cloud" / "iteration_3"
        final_dir.mkdir(parents=True)
        (final_dir / "point_cloud.ply").write_bytes(b"ply")
        (final_dir / "sensitivity.ply").write_bytes(b"torch")
        names = run_omniscene.expected_test_image_names(prepared_scene)
        for iteration, training_time in zip(self.EVAL_ITERATIONS, training_times):
            (model_path / "metrics_{}.txt".format(iteration)).write_text(
                "PSNR : {:.7f}\nSSIM : {:.7f}\nLPIPS : {:.7f}\n".format(
                    20.0 + iteration, 0.7 + iteration / 100.0, 0.2 - iteration / 100.0
                ),
                encoding="utf-8",
            )
            (model_path / "training_time_{}.txt".format(iteration)).write_text(
                "TRAINING_TIME_SECONDS : {:.7f}\n".format(training_time),
                encoding="utf-8",
            )
            for directory_name in ("renders", "gt"):
                output_dir = model_path / "test" / "ours_{}".format(iteration) / directory_name
                output_dir.mkdir(parents=True, exist_ok=True)
                for name in names:
                    Image.new("RGB", (10, 8), color=(iteration, 0, 0)).save(output_dir / name)

    def test_command_is_one_run_without_checkpoints(self):
        command = run_omniscene.build_train_command(
            "/env/python", Path("/prepared/scene"), Path("/result/scene"),
            10000, (1000, 5000, 10000), ("--no_hd",),
        )
        self.assertEqual(command.count(str(run_omniscene.REPO_ROOT / "train.py")), 1)
        self.assertIn("--full_eval_metrics", command)
        self.assertIn("-r", command)
        self.assertEqual(command[command.index("-r") + 1], "1")
        self.assertNotIn("--checkpoint_iterations", command)
        self.assertNotIn("--start_checkpoint", command)

    def test_completion_is_artifact_strict_but_git_loose(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sample = _synthetic_sample()
            scene_name = "001_{}".format(sample["bin_token"])
            prepared_scene = preprocess_scene(
                sample, root / "prepared", scene_name, 1, "center150", root, 0.3
            )
            model_path = root / "results" / scene_name
            self._make_outputs(model_path, prepared_scene)
            run_omniscene._write_scene_completion(
                model_path, "center150", scene_name, sample["bin_token"],
                (8, 10), 3, self.EVAL_ITERATIONS,
            )
            marker_path = model_path / "scene_complete.json"
            marker = json.loads(marker_path.read_text())
            marker.update({"git_commit": "different", "git_branch": "dirty", "git_dirty": True})
            _write_json(marker_path, marker)
            self.assertTrue(
                run_omniscene.scene_complete(
                    model_path, prepared_scene, scene_name, sample["bin_token"],
                    (8, 10), 0.3, "center150", 1, root, 3,
                    self.EVAL_ITERATIONS,
                )
            )

            (model_path / "test" / "ours_2" / "renders" / "007.png").unlink()
            self.assertFalse(
                run_omniscene.scene_complete(
                    model_path, prepared_scene, scene_name, sample["bin_token"],
                    (8, 10), 0.3, "center150", 1, root, 3,
                    self.EVAL_ITERATIONS,
                )
            )

    def test_training_time_must_be_monotonic(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sample = _synthetic_sample()
            scene_name = "001_{}".format(sample["bin_token"])
            prepared_scene = preprocess_scene(
                sample, root / "prepared", scene_name, 1, "center150", root, 0.3
            )
            model_path = root / "results" / scene_name
            self._make_outputs(model_path, prepared_scene, training_times=(1.0, 0.5, 3.0))
            self.assertFalse(
                run_omniscene.scene_outputs_complete(
                    model_path, prepared_scene, sample["bin_token"], (8, 10),
                    0.3, "center150", 1, root, 3, self.EVAL_ITERATIONS,
                )
            )

    def test_protocol_rejects_only_semantic_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            root.mkdir(parents=True, exist_ok=True)
            protocol = {
                "protocol_version": 1,
                "split": "center150",
                "resolution": [112, 200],
                "iterations": 10000,
            }
            run_omniscene.ensure_protocol(root, protocol)
            run_omniscene.ensure_protocol(root, dict(protocol))
            changed = dict(protocol)
            changed["iterations"] = 5000
            with self.assertRaisesRegex(RuntimeError, "different semantic protocol"):
                run_omniscene.ensure_protocol(root, changed)

    def test_aggregate_is_equal_weighted_across_scenes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            records = []
            for index in range(2):
                token = "scene{:032x}_bin001".format(index + 1)
                sample = _synthetic_sample(token=token)
                scene_name = "{:02d}_{}".format(index + 1, token)
                prepared_scene = preprocess_scene(
                    sample, root / "prepared", scene_name, index + 1, "val", root, 0.3
                )
                model_path = root / "results" / scene_name
                self._make_outputs(model_path, prepared_scene)
                if index == 1:
                    for iteration in self.EVAL_ITERATIONS:
                        (model_path / "metrics_{}.txt".format(iteration)).write_text(
                            "PSNR : {:.7f}\nSSIM : {:.7f}\nLPIPS : {:.7f}\n".format(
                                40.0 + iteration, 0.9, 0.1
                            ),
                            encoding="utf-8",
                        )
                run_omniscene._write_scene_completion(
                    model_path, "val", scene_name, token, (8, 10), 3,
                    self.EVAL_ITERATIONS,
                )
                records.append((scene_name, token, prepared_scene, model_path))

            protocol = {
                "protocol_version": 1,
                "split": "val",
                "data_root": os.path.realpath(str(root)),
                "resolution": [8, 10],
                "confidence_threshold": 0.3,
                "n_views": 6,
                "target_views": 18,
                "iterations": 3,
                "eval_iterations": list(self.EVAL_ITERATIONS),
                "extra_train_args": [],
            }
            summary = run_omniscene.aggregate_results(root / "results", records, protocol)
            self.assertAlmostEqual(summary["averages"]["1"]["psnr"], 31.0)
            self.assertEqual(summary["averages"]["3"]["num_samples"], 2)
            self.assertTrue((root / "results" / "val_metrics_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
