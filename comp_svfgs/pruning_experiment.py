"""CPU-only protocol and path checks for the optional pruning experiment."""

import json
from pathlib import Path


EXPERIMENTS = ("no-world-prune",)
LEGACY_RESULTS = "output/omniscene_results"
EXPLORATION_RESULTS = "output/omniscene_revision2"


def paths_overlap(first: Path, second: Path) -> bool:
    first, second = first.resolve(), second.resolve()
    return first == second or first in second.parents or second in first.parents


def resolve_result_root(result_root, prepared_root: Path, data_root: Path, experiment, repo_root: Path) -> Path:
    default = repo_root / EXPLORATION_RESULTS / experiment if experiment else repo_root / LEGACY_RESULTS
    result_root = Path(result_root).resolve() if result_root is not None else default.resolve()
    if experiment:
        protected = [repo_root / LEGACY_RESULTS, repo_root / "output/omniscene_prepared", prepared_root, data_root]
        if any(paths_overlap(result_root, path) for path in protected):
            raise ValueError("Exploration results must not overlap legacy results, shared preprocessing, or source data.")
    elif paths_overlap(result_root, repo_root / EXPLORATION_RESULTS):
        raise ValueError("The exploration results directory requires --experiment.")
    return result_root


def validate_output_children(experiment_root: Path, scene_names, mode: str) -> None:
    names = [mode + suffix for suffix in ("_protocol.json", "_metrics_summary.json", "_metrics_summary.txt")]
    for name in [*names, *scene_names]:
        path = experiment_root / name
        if path.is_symlink() or path.resolve().parent != experiment_root.resolve():
            raise ValueError("Exploration output must stay inside its own directory: {}".format(path))
    for name in scene_names:
        summary = experiment_root / name / "view_metrics_summary.json"
        if summary.is_symlink():
            raise ValueError("Exploration summary must not be a symlink: {}".format(summary))


def experiment_spec(name: str, optimization) -> dict:
    if name not in EXPERIMENTS:
        raise ValueError("Unknown pruning experiment: {}".format(name))
    return {
        "name": name, "world_size_pruning": False, "diagnostics_version": 1,
        **{key: int(optimization[key]) for key in
           ("densify_from_iter", "densify_until_iter", "densification_interval", "opacity_reset_interval")},
    }


def expected_pruning_steps(spec: dict, iterations: int):
    interval = spec["densification_interval"]
    if interval <= 0:
        raise ValueError("Densification interval must be positive.")
    return [step for step in range(interval, iterations + 1, interval)
            if spec["densify_from_iter"] < step < spec["densify_until_iter"]]


def validate_training_experiment(model_path: str, name: str, optimization) -> None:
    spec = json.loads((Path(model_path) / "pruning_experiment.json").read_text())
    if spec != experiment_spec(name, vars(optimization)):
        raise ValueError("Training arguments do not match the scene's pruning experiment protocol.")


def validate_pruning_trace(trace: dict, spec: dict, iterations: int, eval_iterations) -> None:
    if trace.get("pruning_experiment") != spec or trace.get("iterations") != iterations:
        raise ValueError("Training trace lacks the matching pruning experiment.")
    events = trace.get("pruning_events")
    if not isinstance(events, list) or [event.get("iteration") for event in events] != expected_pruning_steps(spec, iterations):
        raise ValueError("Pruning diagnostic event sequence is incomplete.")
    for event in events:
        if event.get("world_pruning_active") is not False:
            raise ValueError("World-size pruning was not disabled in the training trace.")
        if event.get("kind") == "sparse_reinitialization":
            if event["iteration"] % 5000 != 0:
                raise ValueError("Unexpected sparse reinitialization step.")
            continue
        if event.get("kind") != "prune":
            raise ValueError("Unknown pruning event kind.")
        for key in ("points_before_prune", "points_after_prune", "opacity_candidates", "screen_candidates",
                    "world_candidates", "world_only_candidates", "pruned_total"):
            if not isinstance(event.get(key), int) or event[key] < 0:
                raise ValueError("Invalid pruning count: {}".format(key))
        if event["points_before_prune"] - event["pruned_total"] != event["points_after_prune"]:
            raise ValueError("Pruning counts are inconsistent.")
    counts = trace.get("gaussian_count_at_milestones", {})
    if set(counts) != {str(step) for step in eval_iterations} or any(not isinstance(n, int) or n <= 0 for n in counts.values()):
        raise ValueError("Missing evaluated Gaussian counts.")
