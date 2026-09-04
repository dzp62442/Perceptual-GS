#!/usr/bin/env python3
"""Run Perceptual-GS independently on each OmniScene bin."""

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from comp_svfgs.dataset_omniscene import (  # noqa: E402
    CENTER150_SAMPLE_COUNT,
    OmniSceneDataset,
)
from comp_svfgs.omniscene_preprocess import (  # noqa: E402
    PREPARED_FORMAT_VERSION,
    TARGET_VIEW_COUNT,
    TRAIN_VIEW_COUNT,
    prepared_scene_complete,
    preprocess_scene,
)


DEFAULT_ITERATIONS = 10000
DEFAULT_EVAL_ITERATIONS = (1000, 5000, 10000)
DEFAULT_CONFIDENCE_THRESHOLD = 0.3
PROTOCOL_VERSION = 1
SCENE_COMPLETE_VERSION = 1
METRIC_NAMES = ("PSNR", "SSIM", "LPIPS")
TRAINING_TIME_KEY = "training_time_seconds"
RESERVED_EXTRA_ARGS = {
    "-s", "--source_path", "-m", "--model_path", "-r", "--resolution",
    "--eval", "--iterations", "--test_iterations", "--save_iterations",
    "--checkpoint_iterations", "--start_checkpoint", "--full_eval_metrics",
    "--edge_mode",
}


def parse_resolution(value: str) -> Tuple[int, int]:
    try:
        height, width = value.lower().replace(",", "x").split("x")
        resolution = int(height), int(width)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("Resolution must be HxW, for example 112x200") from exc
    if resolution not in ((112, 200), (224, 400)):
        raise argparse.ArgumentTypeError("Resolution must be 112x200 or 224x400")
    return resolution


def _absolute_path(path: os.PathLike) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _atomic_write_json(path: Path, payload: Dict) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    os.replace(str(temporary_path), str(path))


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(str(temporary_path), str(path))


def parse_metrics(path: Path) -> Dict[str, float]:
    values = {}
    with path.open("r", encoding="utf-8") as metrics_file:
        for line in metrics_file:
            name, separator, value = line.partition(":")
            name = name.strip()
            if separator and name in METRIC_NAMES:
                values[name] = float(value.strip())
    if set(values) != set(METRIC_NAMES):
        raise ValueError("Incomplete metric file: {}".format(path))
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite metric in {}".format(path))
    return values


def parse_training_time(path: Path) -> float:
    name, separator, value = path.read_text(encoding="utf-8").strip().partition(":")
    if not separator or name.strip() != "TRAINING_TIME_SECONDS":
        raise ValueError("Invalid training-time file: {}".format(path))
    training_time = float(value.strip())
    if not math.isfinite(training_time) or training_time < 0.0:
        raise ValueError("Invalid training time in {}".format(path))
    return training_time


def _artifact_exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _readable_png(path: Path, resolution: Optional[Tuple[int, int]] = None) -> bool:
    if not _artifact_exists(path):
        return False
    try:
        with Image.open(path) as image:
            if resolution is not None and image.size != (resolution[1], resolution[0]):
                return False
            image.verify()
    except (OSError, ValueError):
        return False
    return True


def expected_test_image_names(prepared_scene: Path) -> List[str]:
    transforms_path = prepared_scene / "transforms_test.json"
    with transforms_path.open("r", encoding="utf-8") as transforms_file:
        frames = json.load(transforms_file).get("frames", [])
    names = [Path(frame["file_path"]).stem + ".png" for frame in frames]
    if len(names) != TARGET_VIEW_COUNT or len(set(names)) != TARGET_VIEW_COUNT:
        raise ValueError("Expected exactly 18 unique test views in {}".format(transforms_path))
    return names


def iteration_complete(model_path: Path, prepared_scene: Path, iteration: int) -> bool:
    try:
        parse_metrics(model_path / "metrics_{}.txt".format(iteration))
        parse_training_time(model_path / "training_time_{}.txt".format(iteration))
        expected_names = expected_test_image_names(prepared_scene)
        meta = json.loads((prepared_scene / "meta.json").read_text(encoding="utf-8"))
        resolution = tuple(int(value) for value in meta["resolution"])
        if len(resolution) != 2:
            return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

    iteration_dir = model_path / "test" / "ours_{}".format(iteration)
    for directory_name in ("renders", "gt"):
        directory = iteration_dir / directory_name
        if not directory.is_dir():
            return False
        paths = sorted(directory.glob("*.png"))
        if [path.name for path in paths] != sorted(expected_names):
            return False
        if not all(_readable_png(path, resolution) for path in paths):
            return False
    return True


def _scene_marker_matches(
    marker_path: Path,
    mode: str,
    scene_name: str,
    bin_token: str,
    resolution: Tuple[int, int],
    iterations: int,
    eval_iterations: Sequence[int],
) -> bool:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        return (
            marker.get("state_version") == SCENE_COMPLETE_VERSION
            and marker.get("protocol_version") == PROTOCOL_VERSION
            and marker.get("split") == mode
            and marker.get("scene_name") == scene_name
            and marker.get("bin_token") == bin_token
            and marker.get("resolution") == list(resolution)
            and marker.get("iterations") == int(iterations)
            and marker.get("eval_iterations") == list(eval_iterations)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def scene_outputs_complete(
    model_path: Path,
    prepared_scene: Path,
    bin_token: str,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    mode: str,
    scene_index: int,
    data_root: os.PathLike,
    iterations: int,
    eval_iterations: Sequence[int],
) -> bool:
    if not prepared_scene_complete(
        prepared_scene,
        bin_token,
        resolution,
        confidence_threshold,
        mode=mode,
        data_root=data_root,
        scene_index=scene_index,
    ):
        return False
    if not all(iteration_complete(model_path, prepared_scene, iteration) for iteration in eval_iterations):
        return False
    required_files = (
        model_path / "cfg_args",
        model_path / "input.ply",
        model_path / "cameras.json",
        model_path / "point_cloud" / "iteration_{}".format(iterations) / "point_cloud.ply",
        model_path / "point_cloud" / "iteration_{}".format(iterations) / "sensitivity.ply",
    )
    if any(not _artifact_exists(path) for path in required_files):
        return False
    if list(model_path.glob("chkpnt*.pth")):
        return False
    try:
        training_times = [
            parse_training_time(model_path / "training_time_{}.txt".format(iteration))
            for iteration in eval_iterations
        ]
    except (OSError, ValueError):
        return False
    return training_times == sorted(training_times)


def scene_complete(
    model_path: Path,
    prepared_scene: Path,
    scene_name: str,
    bin_token: str,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    mode: str,
    scene_index: int,
    data_root: os.PathLike,
    iterations: int,
    eval_iterations: Sequence[int],
) -> bool:
    if not scene_outputs_complete(
        model_path, prepared_scene, bin_token, resolution, confidence_threshold,
        mode, scene_index, data_root, iterations, eval_iterations,
    ):
        return False
    return _scene_marker_matches(
        model_path / "scene_complete.json", mode, scene_name, bin_token,
        resolution, iterations, eval_iterations,
    )


def _validate_extra_train_args(parser: argparse.ArgumentParser, extra_args: Sequence[str]) -> None:
    for value in extra_args:
        option = value.split("=", 1)[0]
        attached_reserved_short = any(
            option.startswith(short_option) and option != short_option
            for short_option in ("-s", "-m", "-r")
        )
        if option in RESERVED_EXTRA_ARGS or attached_reserved_short:
            parser.error(
                "{} is managed by run_omniscene.py and cannot appear in --extra-train-args".format(
                    option
                )
            )


def build_train_command(
    python_executable: str,
    prepared_scene: Path,
    model_path: Path,
    iterations: int,
    eval_iterations: Sequence[int],
    extra_train_args: Sequence[str],
) -> List[str]:
    command = [
        python_executable,
        str(REPO_ROOT / "train.py"),
        "--eval",
        "-s", str(prepared_scene),
        "-m", str(model_path),
        "-r", "1",
        "--iterations", str(iterations),
        "--test_iterations", *[str(value) for value in eval_iterations],
        "--save_iterations", str(iterations),
        "--full_eval_metrics",
    ]
    command.extend(extra_train_args)
    return command


def _run_command(command: Sequence[str], gpu: str) -> None:
    print("[RUN] {}".format(" ".join(shlex.quote(value) for value in command)), flush=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(command, check=True, cwd=str(REPO_ROOT), env=environment)


def _write_scene_completion(
    model_path: Path,
    mode: str,
    scene_name: str,
    bin_token: str,
    resolution: Tuple[int, int],
    iterations: int,
    eval_iterations: Sequence[int],
) -> None:
    metrics = {}
    for iteration in eval_iterations:
        values = parse_metrics(model_path / "metrics_{}.txt".format(iteration))
        metrics[str(iteration)] = {
            "psnr": values["PSNR"],
            "ssim": values["SSIM"],
            "lpips": values["LPIPS"],
            TRAINING_TIME_KEY: parse_training_time(
                model_path / "training_time_{}.txt".format(iteration)
            ),
        }
    _atomic_write_json(
        model_path / "scene_complete.json",
        {
            "state_version": SCENE_COMPLETE_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "prepared_format_version": PREPARED_FORMAT_VERSION,
            "split": mode,
            "scene_name": scene_name,
            "bin_token": bin_token,
            "resolution": list(resolution),
            "iterations": int(iterations),
            "eval_iterations": list(eval_iterations),
            "metrics": metrics,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _safe_remove_result_scene(model_path: Path, experiment_root: Path) -> None:
    if model_path.is_symlink():
        raise RuntimeError("Refusing to remove a symlinked result directory: {}".format(model_path))
    if model_path.resolve().parent != experiment_root.resolve():
        raise RuntimeError("Refusing to remove a result outside experiment root: {}".format(model_path))
    shutil.rmtree(str(model_path))


def run_scene(
    dataset: OmniSceneDataset,
    dataset_index: int,
    prepared_root: Path,
    experiment_root: Path,
    confidence_threshold: float,
    gpu: str,
    iterations: int,
    eval_iterations: Sequence[int],
    extra_train_args: Sequence[str],
) -> Tuple[str, str, Path, Path]:
    scene_index = dataset_index + 1
    scene_name = dataset.scene_name(dataset_index)
    bin_token = dataset.bin_tokens[dataset_index]
    prepared_scene = prepared_root / scene_name
    model_path = experiment_root / scene_name
    record = scene_name, bin_token, prepared_scene, model_path

    # This check deliberately does not inspect source hashes, Git commit/branch,
    # or clean/dirty state. Durable artifacts and semantic protocol are enough.
    if scene_complete(
        model_path, prepared_scene, scene_name, bin_token, dataset.resolution,
        confidence_threshold, dataset.mode, scene_index, dataset.data_root,
        iterations, eval_iterations,
    ):
        print("[SKIP] Completed scene: {}".format(scene_name), flush=True)
        return record

    scene_data = dataset[dataset_index]
    prepared_scene = preprocess_scene(
        scene_data=scene_data,
        output_root=prepared_root,
        scene_name=scene_name,
        scene_index=scene_index,
        mode=dataset.mode,
        data_root=dataset.data_root,
        confidence_threshold=confidence_threshold,
    )
    record = scene_name, bin_token, prepared_scene, model_path

    if model_path.exists():
        print("[RESTART] Incomplete scene, restarting from zero: {}".format(scene_name), flush=True)
        _safe_remove_result_scene(model_path, experiment_root)
    else:
        print("[START] Training from zero: {}".format(scene_name), flush=True)
    model_path.mkdir(parents=True, exist_ok=False)

    command = build_train_command(
        sys.executable, prepared_scene, model_path, iterations,
        eval_iterations, extra_train_args,
    )
    _atomic_write_json(model_path / "runner_command.json", {"command": command})
    _run_command(command, gpu)

    if not scene_outputs_complete(
        model_path, prepared_scene, bin_token, dataset.resolution,
        confidence_threshold, dataset.mode, scene_index, dataset.data_root,
        iterations, eval_iterations,
    ):
        raise RuntimeError("Scene did not produce all required artifacts: {}".format(scene_name))
    _write_scene_completion(
        model_path, dataset.mode, scene_name, bin_token, dataset.resolution,
        iterations, eval_iterations,
    )
    if not scene_complete(
        model_path, prepared_scene, scene_name, bin_token, dataset.resolution,
        confidence_threshold, dataset.mode, scene_index, dataset.data_root,
        iterations, eval_iterations,
    ):
        raise RuntimeError("Scene completion marker failed validation: {}".format(scene_name))
    print("[DONE] {}".format(scene_name), flush=True)
    return record


def aggregate_results(
    experiment_root: Path,
    scene_records: Sequence[Tuple[str, str, Path, Path]],
    protocol: Dict,
) -> Dict:
    expected_count = len(scene_records)
    if protocol["split"] == "center150" and expected_count != CENTER150_SAMPLE_COUNT:
        raise RuntimeError("Center150 aggregation requires exactly 150 scene records")

    eval_iterations = tuple(protocol["eval_iterations"])
    accumulators = {
        iteration: {name: [] for name in METRIC_NAMES + (TRAINING_TIME_KEY,)}
        for iteration in eval_iterations
    }
    samples = []
    for scene_index, (scene_name, bin_token, prepared_scene, model_path) in enumerate(
        scene_records, 1
    ):
        if not scene_complete(
            model_path, prepared_scene, scene_name, bin_token,
            tuple(protocol["resolution"]), protocol["confidence_threshold"],
            protocol["split"], scene_index, protocol["data_root"],
            protocol["iterations"], eval_iterations,
        ):
            raise RuntimeError("Cannot aggregate incomplete scene: {}".format(scene_name))
        sample_metrics = {}
        for iteration in eval_iterations:
            values = parse_metrics(model_path / "metrics_{}.txt".format(iteration))
            training_time = parse_training_time(
                model_path / "training_time_{}.txt".format(iteration)
            )
            sample_metrics[str(iteration)] = {
                "psnr": values["PSNR"],
                "ssim": values["SSIM"],
                "lpips": values["LPIPS"],
                TRAINING_TIME_KEY: training_time,
            }
            for name in METRIC_NAMES:
                accumulators[iteration][name].append(values[name])
            accumulators[iteration][TRAINING_TIME_KEY].append(training_time)
        samples.append(
            {"scene_name": scene_name, "bin_token": bin_token, "metrics": sample_metrics}
        )

    averages = {}
    for iteration in eval_iterations:
        averages[str(iteration)] = {
            "num_samples": expected_count,
            "psnr": sum(accumulators[iteration]["PSNR"]) / expected_count,
            "ssim": sum(accumulators[iteration]["SSIM"]) / expected_count,
            "lpips": sum(accumulators[iteration]["LPIPS"]) / expected_count,
            TRAINING_TIME_KEY: (
                sum(accumulators[iteration][TRAINING_TIME_KEY]) / expected_count
            ),
        }

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "split": protocol["split"],
        "sample_count": expected_count,
        "data_root": protocol["data_root"],
        "resolution": protocol["resolution"],
        "confidence_threshold": protocol["confidence_threshold"],
        "iterations": protocol["iterations"],
        "eval_iterations": protocol["eval_iterations"],
        "extra_train_args": protocol["extra_train_args"],
        "averages": averages,
        "samples": samples,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    prefix = "center150" if protocol["split"] == "center150" else protocol["split"]
    _atomic_write_json(experiment_root / "{}_metrics_summary.json".format(prefix), summary)

    lines = ["{} samples: {}".format(prefix, expected_count)]
    for iteration in eval_iterations:
        result = averages[str(iteration)]
        lines.extend(
            (
                "Iteration {}".format(iteration),
                "PSNR : {:.7f}".format(result["psnr"]),
                "SSIM : {:.7f}".format(result["ssim"]),
                "LPIPS : {:.7f}".format(result["lpips"]),
                "TRAINING_TIME_SECONDS : {:.7f}".format(result[TRAINING_TIME_KEY]),
            )
        )
    _atomic_write_text(
        experiment_root / "{}_metrics_summary.txt".format(prefix), "\n".join(lines) + "\n"
    )
    print("[SUMMARY] {} scenes".format(expected_count), flush=True)
    return summary


def _protocol_tag(
    mode: str,
    resolution: Tuple[int, int],
    confidence_threshold: float,
    iterations: int,
    eval_iterations: Sequence[int],
    extra_train_args: Sequence[str],
) -> str:
    tag = "{}_{}x{}".format(mode, resolution[0], resolution[1])
    if iterations != DEFAULT_ITERATIONS or tuple(eval_iterations) != DEFAULT_EVAL_ITERATIONS:
        tag += "_i{}_e{}".format(iterations, "-".join(str(value) for value in eval_iterations))
    if not math.isclose(confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD, abs_tol=1e-12):
        tag += "_conf{:g}".format(confidence_threshold)
    if extra_train_args:
        digest = hashlib.sha256("\0".join(extra_train_args).encode("utf-8")).hexdigest()[:8]
        tag += "_args{}".format(digest)
    return tag


def _prepared_tag(mode: str, resolution: Tuple[int, int], confidence_threshold: float) -> str:
    tag = "{}_{}x{}".format(mode, resolution[0], resolution[1])
    if not math.isclose(confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD, abs_tol=1e-12):
        tag += "_conf{:g}".format(confidence_threshold)
    return tag


def ensure_protocol(experiment_root: Path, protocol: Dict) -> None:
    protocol_path = experiment_root / "{}_protocol.json".format(protocol["split"])
    if protocol_path.is_file():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(
                "Result directory contains a different semantic protocol: {}".format(protocol_path)
            )
        return
    existing_entries = list(experiment_root.iterdir())
    if existing_entries:
        raise RuntimeError(
            "Refusing to adopt a non-empty result directory without a protocol file: {}".format(
                experiment_root
            )
        )
    _atomic_write_json(protocol_path, protocol)


def _all_scene_records(dataset: OmniSceneDataset, prepared_root: Path, experiment_root: Path):
    return [
        (
            dataset.scene_name(index),
            token,
            prepared_root / dataset.scene_name(index),
            experiment_root / dataset.scene_name(index),
        )
        for index, token in enumerate(dataset.bin_tokens)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OmniScene per-bin preprocessing, optimization, evaluation, and aggregation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default="datasets/omniscene")
    parser.add_argument("--prepared-root", default="output/omniscene_prepared")
    parser.add_argument("--result-root", default="output/omniscene_results")
    parser.add_argument(
        "--mode", choices=("train", "val", "test", "demo", "center150"),
        default="center150",
    )
    parser.add_argument("--resolution", type=parse_resolution, default=(112, 200))
    parser.add_argument("--n-views", type=int, default=TRAIN_VIEW_COUNT)
    parser.add_argument("--conf-threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--eval-iterations", nargs="+", type=int,
        default=list(DEFAULT_EVAL_ITERATIONS),
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--scene-indices", nargs="+", type=int,
        help="Optional one-based scene indices; useful for smoke tests and partial scheduling",
    )
    parser.add_argument(
        "--extra-train-args", nargs=argparse.REMAINDER, default=[],
        help="Arguments passed through to train.py; this option must appear last",
    )
    args = parser.parse_args()

    if args.n_views != TRAIN_VIEW_COUNT:
        parser.error("--n-views is fixed to 6 for this protocol")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if not 0.0 <= args.conf_threshold <= 1.0:
        parser.error("--conf-threshold must be in [0, 1]")
    eval_iterations = tuple(args.eval_iterations)
    if (
        not eval_iterations
        or any(value <= 0 for value in eval_iterations)
        or tuple(sorted(set(eval_iterations))) != eval_iterations
    ):
        parser.error("--eval-iterations must be unique, positive, and strictly increasing")
    if eval_iterations[-1] != args.iterations:
        parser.error("The final --eval-iterations value must equal --iterations")
    _validate_extra_train_args(parser, args.extra_train_args)

    data_root = _absolute_path(args.data_root)
    prepared_base = _absolute_path(args.prepared_root)
    result_base = _absolute_path(args.result_root)
    dataset = OmniSceneDataset(data_root, mode=args.mode, resolution=args.resolution)
    prepared_root = prepared_base / _prepared_tag(
        args.mode, args.resolution, args.conf_threshold
    )
    experiment_root = result_base / _protocol_tag(
        args.mode, args.resolution, args.conf_threshold, args.iterations,
        eval_iterations, args.extra_train_args,
    )
    prepared_root.mkdir(parents=True, exist_ok=True)
    experiment_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "split": args.mode,
        "data_root": os.path.realpath(str(data_root)),
        "resolution": list(args.resolution),
        "confidence_threshold": float(args.conf_threshold),
        "n_views": TRAIN_VIEW_COUNT,
        "target_views": TARGET_VIEW_COUNT,
        "iterations": int(args.iterations),
        "eval_iterations": list(eval_iterations),
        "extra_train_args": list(args.extra_train_args),
    }
    ensure_protocol(experiment_root, protocol)

    if args.scene_indices is None:
        selected_indices = list(range(len(dataset)))
    else:
        if len(set(args.scene_indices)) != len(args.scene_indices):
            parser.error("--scene-indices cannot contain duplicates")
        if any(value < 1 or value > len(dataset) for value in args.scene_indices):
            parser.error("--scene-indices values must be in [1, {}]".format(len(dataset)))
        selected_indices = [value - 1 for value in args.scene_indices]

    for dataset_index in selected_indices:
        run_scene(
            dataset, dataset_index, prepared_root, experiment_root,
            args.conf_threshold, args.gpu, args.iterations, eval_iterations,
            args.extra_train_args,
        )

    all_records = _all_scene_records(dataset, prepared_root, experiment_root)
    all_complete = all(
        scene_complete(
            model_path, prepared_scene, scene_name, bin_token, dataset.resolution,
            args.conf_threshold, dataset.mode, index, dataset.data_root,
            args.iterations, eval_iterations,
        )
        for index, (scene_name, bin_token, prepared_scene, model_path) in enumerate(
            all_records, 1
        )
    )
    if all_complete:
        aggregate_results(experiment_root, all_records, protocol)
    else:
        print(
            "[PARTIAL] Selected scenes finished, but not all {} scenes are complete; summary not written.".format(
                len(dataset)
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
