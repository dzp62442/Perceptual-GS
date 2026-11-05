"""CPU regression checks for isolation, sample recovery, and the pruning intervention."""

import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from comp_svfgs.pruning_experiment import (
    experiment_spec, expected_pruning_steps, resolve_result_root,
    validate_output_children, validate_pruning_trace,
)
from scripts import run_omniscene as runner
import test_omniscene_protocol as fixtures


SPEC = experiment_spec("no-world-prune", {
    "densify_from_iter": 500, "densify_until_iter": 15000,
    "densification_interval": 100, "opacity_reset_interval": 3000,
})


class PruningExperimentTest(unittest.TestCase):
    def test_default_command_and_opt_in_preserve_budget_and_retention(self):
        args = runner.parse_args([])
        self.assertIsNone(args.experiment)
        self.assertEqual(args.iterations, 10000)
        common = ("python", Path("/cache/scene"), Path("/result/scene"), 10000, (1000, 5000, 10000), [])
        old, new = runner.build_train_command(*common), runner.build_train_command(*common, SPEC)
        self.assertEqual(new, old + ["--pruning_experiment", "no-world-prune"])
        self.assertEqual(old[old.index("--save_iterations") + 1], "10000")
        self.assertNotIn("--checkpoint_iterations", new)
        self.assertNotIn("--start_checkpoint", new)
        for option in ("--pruning_experiment", "--pruning_exp", "--save_iter", "--start_check"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["--extra-train-args", option, "anything"])

    def test_results_cannot_overlap_old_results_cache_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, data = root / "output/omniscene_prepared", root / "data"
            self.assertEqual(resolve_result_root(None, prepared, data, None, root), root / "output/omniscene_results")
            exploration = resolve_result_root(None, prepared, data, "no-world-prune", root)
            self.assertEqual(exploration, root / "output/omniscene_revision2/no-world-prune")
            for path in [root, root / "output/omniscene_results", prepared / "child", data]:
                with self.assertRaises(ValueError):
                    resolve_result_root(path, prepared, data, "no-world-prune", root)
            old = root / "output/omniscene_results"
            old.mkdir(parents=True)
            alias = root / "alias"
            alias.symlink_to(old, target_is_directory=True)
            with self.assertRaises(ValueError):
                resolve_result_root(alias, prepared, data, "no-world-prune", root)
            exploration.mkdir(parents=True)
            (exploration / "001_scene").symlink_to(old, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_output_children(exploration, ["001_scene"], "center150")

    def test_protocol_cannot_mix_modes_and_dry_run_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new"
            protocol = {"split": "center150", "iterations": 10000, "pruning_experiment": SPEC}
            runner.ensure_protocol(root, protocol, dry_run=True)
            self.assertFalse(root.exists())
            root.mkdir()
            runner.ensure_protocol(root, protocol)
            with self.assertRaisesRegex(RuntimeError, "different semantic protocol"):
                runner.ensure_protocol(root, {"split": "center150", "iterations": 10000}, dry_run=True)

    def _fixture(self, root):
        sample = fixtures._synthetic_sample()
        name = "001_" + sample["bin_token"]
        prepared = runner.preprocess_scene(sample, root / "prepared", name, 1, "val", root, 0.3)
        dataset = SimpleNamespace(bin_tokens=[sample["bin_token"]], resolution=(8, 10), mode="val",
                                  data_root=root, scene_name=lambda index: name)
        return dataset, name, prepared, root / "results" / name

    def _outputs(self, model, prepared):
        fixtures.RunnerProtocolTest()._make_outputs(model, prepared)
        (model / "pruning_experiment.json").write_text(json.dumps(SPEC))
        (model / "omniscene_training_trace.json").write_text(json.dumps({
            "pruning_experiment": SPEC, "iterations": 3, "pruning_events": [],
            "gaussian_count_at_milestones": {str(i): 10 for i in (1, 2, 3)},
        }))

    def test_complete_exploration_skips_training_and_default_cannot_accept_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, name, prepared, model = self._fixture(root)
            self._outputs(model, prepared)
            runner._write_scene_completion(model, "val", name, dataset.bin_tokens[0], (8, 10), 3, (1, 2, 3), SPEC)
            with mock.patch.object(runner, "preprocess_scene", side_effect=AssertionError("must reuse cache")), \
                 mock.patch.object(runner, "_run_command", side_effect=AssertionError("must skip training")):
                runner.run_scene(dataset, 0, prepared.parent, model.parent, 0.3, "0", 3, (1, 2, 3), [], SPEC)
            common = (model, prepared, name, dataset.bin_tokens[0], (8, 10), 0.3, "val", 1, root, 3, (1, 2, 3))
            self.assertFalse(runner.scene_complete(*common))
            self.assertTrue(runner.scene_complete(*common, SPEC))
            (model / "point_cloud/iteration_1").mkdir()
            self.assertFalse(runner.scene_complete(*common, SPEC))

    def test_incomplete_sample_restarts_without_reading_raw_data_or_rebuilding_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, name, prepared, model = self._fixture(root)
            model.mkdir(parents=True)
            (model / "chkpnt1000.pth").write_bytes(b"obsolete")
            def fake_train(command, gpu):
                self.assertFalse((model / "chkpnt1000.pth").exists())
                self.assertNotIn("--start_checkpoint", command)
                self._outputs(model, prepared)
            with mock.patch.object(runner, "preprocess_scene", side_effect=AssertionError("must reuse cache")), \
                 mock.patch.object(runner, "_run_command", side_effect=fake_train) as run:
                runner.run_scene(dataset, 0, prepared.parent, model.parent, 0.3, "0", 3, (1, 2, 3), [], SPEC)
                self.assertEqual(run.call_count, 1)
            self.assertTrue((model / "scene_complete.json").exists())

    def test_missing_cache_fails_before_removing_partial_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, name, prepared, model = self._fixture(root)
            (prepared / "points3d.ply").unlink()
            model.mkdir(parents=True)
            sentinel = model / "keep"
            sentinel.write_text("incomplete")
            with mock.patch.object(runner, "preprocess_scene") as prepare, mock.patch.object(runner, "_run_command") as run:
                with self.assertRaisesRegex(RuntimeError, "never rebuilds"):
                    runner.run_scene(dataset, 0, prepared.parent, model.parent, 0.3, "0", 3, (1, 2, 3), [], SPEC)
                prepare.assert_not_called()
                run.assert_not_called()
            self.assertTrue(sentinel.exists())

    def test_trace_accepts_native_sparse_reinitialization_but_rejects_missing_events(self):
        steps = expected_pruning_steps(SPEC, 10000)
        events = [{"iteration": step, "kind": "prune", "world_pruning_active": False,
                   "points_before_prune": 10, "points_after_prune": 9, "pruned_total": 1,
                   "opacity_candidates": 1, "screen_candidates": 0, "world_candidates": 3,
                   "world_only_candidates": 3} for step in steps]
        for i, event in enumerate(events):
            if event["iteration"] % 5000 == 0:
                events[i] = {"iteration": event["iteration"], "kind": "sparse_reinitialization", "world_pruning_active": False}
        trace = {"pruning_experiment": SPEC, "iterations": 10000, "pruning_events": events,
                 "gaussian_count_at_milestones": {str(i): 10 for i in (1000, 5000, 10000)}}
        validate_pruning_trace(trace, SPEC, 10000, (1000, 5000, 10000))
        trace["pruning_events"] = events[:-1]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_pruning_trace(trace, SPEC, 10000, (1000, 5000, 10000))

    def test_world_pruning_switch_keeps_opacity_and_screen_pruning(self):
        source = Path(runner.REPO_ROOT) / "scene/gaussian_model.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GaussianModel")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "densify_and_prune")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        for enabled, screen_size, expected in [(True, 20, [True, True, True, False]),
                                               (False, 20, [True, True, False, False]),
                                               (False, None, [True, False, False, False])]:
            model = SimpleNamespace(xyz_gradient_accum=torch.zeros(4, 1), denom=torch.ones(4, 1),
                                    get_opacity=torch.tensor([[.001], [.9], [.9], [.9]]),
                                    get_scaling=torch.tensor([[.01]*3, [.01]*3, [.2]*3, [.01]*3]),
                                    max_radii2D=torch.tensor([0., 21., 0., 0.]), get_xyz=torch.zeros(4, 3),
                                    densify_and_clone=lambda *a, **k: None, densify_and_split=lambda *a, **k: None)
            def prune(mask):
                self.assertEqual(mask.tolist(), expected)
                model.get_xyz = model.get_xyz[~mask]
            model.prune_points = prune
            stats = {}
            with mock.patch("torch.cuda.empty_cache"):
                namespace["densify_and_prune"](model, .0002, .005, 1., screen_size, 3100,
                                               SimpleNamespace(hd_interval=1000, md_interval=1500),
                                               False, False, False, 1.2, world_size_pruning=enabled, pruning_stats=stats)
            self.assertEqual(stats["pruned_total"], sum(expected))
            self.assertEqual(stats["world_only_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
