"""Rank the saved Arena candidates with independently calibrated step overhead."""

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean

from compare import read_json


CASES = ("tp2", "pp2_uniform", "pp2_auto")
CALIBRATION_STEPS = 5
INPUTS = ("candidates.json", "selection.json", "profile-source.txt", "vocab-communication.json")
WORKLOAD = (
    "--batch-size", "8", "--seq-len", "256", "--num-layers", "4", "--hidden-size", "4096",
    "--num-attention-heads", "32", "--vocab-size", "50000", "--num-micro-batches", "4",
)


def write_result(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def previous_run(bundle, args):
    if args:
        return Path(args[0]).resolve(strict=True)
    runs = [
        path for path in (bundle / "results").glob("run.*")
        if (path / "summary.json").is_file() and all((path / name).is_file() for name in INPUTS)
    ]
    if not runs:
        raise RuntimeError("No previous complete Test 5 run found under results/run.*.")
    return max(runs, key=lambda path: (path / "summary.json").stat().st_mtime_ns)


def read_candidates(source):
    scores = read_json(source / "candidates.json")
    candidates = {plan["case"]: plan for plan in scores}
    if len(scores) != len(CASES) or set(candidates) != set(CASES):
        raise RuntimeError(f"Expected three Test 5 candidates in {source}.")
    for label, plan in candidates.items():
        score = plan["estimated_forward_backward_ms"]
        degrees, layers = plan["tp_pp_dp"], plan["stage_layers"]
        if label == "tp2":
            valid_layout = degrees == [2, 1, 1] and layers == [4]
        else:
            valid_layers = ([2, 2],) if label == "pp2_uniform" else ([3, 1], [1, 3])
            valid_layout = degrees == [1, 2, 1] and layers in valid_layers
        if not valid_layout or not isinstance(score, (float, int)) or not 0 < score < math.inf:
            raise RuntimeError(f"Invalid candidate in {source}: {plan}.")

    selected = read_json(source / "selection.json")["after_vocab"]
    label = selected["case"]
    if label not in candidates or selected != candidates[label]:
        raise RuntimeError(f"The saved selection does not match the candidate in {source}.")
    if selected["estimated_forward_backward_ms"] != min(p["estimated_forward_backward_ms"] for p in scores):
        raise RuntimeError(f"The saved selection does not have the lowest score in {source}.")
    return candidates, label


def calibration_windows(path, label):
    traces = [read_json(path.with_name(f"{path.stem}.rank{rank}.json")) for rank in (0, 1)]
    for rank, trace in enumerate(traces):
        steps = trace["steps_ns"]
        if (trace["case"], trace["rank"], len(steps)) != (label, rank, CALIBRATION_STEPS):
            raise RuntimeError(f"Invalid calibration count or rank for {label}.")
        for start, forward_start, backward_end, end in steps:
            if not 0 < start <= forward_start < backward_end <= end:
                raise RuntimeError(f"Invalid calibration boundaries for {label} rank {rank}.")
    if traces[0]["host"] != traces[1]["host"]:
        raise RuntimeError("Calibration timestamps must come from the same host.")

    before, after = [], []
    for steps in zip(*(trace["steps_ns"] for trace in traces), strict=True):
        start, forward_start, backward_end, end = zip(*steps)
        # Count the time outside the shared forward/backward window once.
        before.append((min(forward_start) - min(start)) / 1_000_000)
        after.append((max(end) - max(backward_end)) / 1_000_000)
    return mean(before), mean(after)


def calibrate(bundle, output, label, plan):
    plan_path = output / f"{label}-plan.json"
    write_result(plan_path, {**plan, "sequence_length": 256})
    result_path = output / f"calibration-{label}.json"
    log_path = output / f"calibration-{label}.log"
    print(f"TEST5_ASCEND_FULL_STEP_CALIBRATE {label} {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            [
                "timeout", "-k", "15s", "900s", sys.executable, "-u", "-m", "torch.distributed.run",
                "--standalone", "--nnodes=1", "--nproc-per-node=2",
                str(bundle / "scripts/full_step_calibrate.py"), "timed", label, "0",
                str(plan_path), str(result_path), *WORKLOAD,
            ],
            cwd=bundle / "source/crius-torch", stdout=log, stderr=subprocess.STDOUT,
        )
    if process.returncode:
        raise RuntimeError(f"Calibration exited {process.returncode} for {label}. See {log_path}.")
    before, after = calibration_windows(result_path, label)
    return {
        "case": label,
        "tp_pp_dp": plan["tp_pp_dp"],
        "stage_layers": plan["stage_layers"],
        "estimated_forward_backward_ms": plan["estimated_forward_backward_ms"],
        "calibrated_before_forward_backward_ms": before,
        "calibrated_after_forward_backward_ms": after,
        "estimated_full_step_ms": plan["estimated_forward_backward_ms"] + before + after,
    }


def main():
    bundle = Path(__file__).resolve().parents[1]
    source = previous_run(bundle, sys.argv[1:])
    candidates, original_selection = read_candidates(source)
    output = Path(tempfile.mkdtemp(prefix="full-step.", dir=bundle / "results"))
    print(f"TEST5_ASCEND_FULL_STEP_DIR {output}", flush=True)
    print(f"TEST5_ASCEND_FULL_STEP_SOURCE {source}", flush=True)
    for name in INPUTS:
        shutil.copyfile(source / name, output / name)

    scored = [calibrate(bundle, output, label, candidates[label]) for label in CASES]
    predicted = min(scored, key=lambda plan: plan["estimated_full_step_ms"])["case"]
    prediction = {
        "source_run": str(source),
        "calibration_steps_per_layout": CALIBRATION_STEPS,
        "original_forward_backward_selection": original_selection,
        "full_step_prediction_selection": predicted,
        "candidate_scores": scored,
    }
    write_result(output / "full-step-prediction.json", prediction)
    print("TEST5_ASCEND_FULL_STEP_PREDICTION", json.dumps(prediction), flush=True)

    comparison = subprocess.run(
        [sys.executable, "-u", str(bundle / "scripts/compare.py"), str(output), "--num-nodes", "1", *WORKLOAD],
        cwd=bundle / "source/crius-torch",
    )
    if comparison.returncode:
        raise RuntimeError(f"The independent timing run exited {comparison.returncode}. See {output}.")

    holdout = read_json(output / "summary.json")
    measured = holdout["measured_layouts"]
    fastest = holdout["fastest_measured_layout"]
    if set(measured) != set(CASES) or fastest not in measured or holdout["runtime_failures"]:
        raise RuntimeError(f"The independent timing run did not measure all three candidates. See {output}.")
    summary = {
        **prediction,
        "holdout_rounds": holdout["rounds"],
        "holdout_timed_steps_per_round": holdout["timed_steps_per_run"],
        "fastest_measured_layout": fastest,
        "original_selection_matches_fastest": original_selection == fastest,
        "full_step_prediction_matches_fastest": predicted == fastest,
        "measured_layouts": measured,
    }
    write_result(output / "full-step-summary.json", summary)
    print("TEST5_ASCEND_FULL_STEP_COMPARISON", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
