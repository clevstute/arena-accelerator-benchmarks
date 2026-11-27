import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def launch(output, mode, label, pair, plan_file, result_file, workload):
    log_name = f"verify-{label}.log" if mode == "verify" else f"{pair}-{label}.log"
    with (output / log_name).open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                "timeout", "-k", "15s", "900s", sys.executable, "-u", "-m", "torch.distributed.run",
                "--standalone", "--nnodes=1", "--nproc-per-node=2",
                str(Path(__file__).with_name("replay.py")), mode, label, str(pair), str(plan_file), str(result_file),
                *workload,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"Training exited {completed.returncode}; see {log_name}.")


def check_result(run, label, pair, plan):
    actual = (
        run["case"], run["round"], run["sequence_length"], run["tokens_per_step"],
        run["tp_pp_dp"], run["stage_layers"], run["optimizer_updates"],
    )
    if actual != (label, pair, 256, 2048, plan["tp_pp_dp"], plan["stage_layers"], 7):
        raise RuntimeError(f"Wrong training workload or incomplete weight updates for {label}: {run}.")
    full, forward_backward = run["step_seconds"], run["forward_backward_seconds"]
    if len(full) != 5 or len(forward_backward) != 5 or not all(
        0 < fb <= step < math.inf for step, fb in zip(full, forward_backward, strict=True)
    ):
        raise RuntimeError(f"Invalid timing samples for {label}: {run}.")


def summarize(plan, runs):
    step_seconds = mean(value for run in runs for value in run["step_seconds"])
    return {
        "tp_pp_dp": plan["tp_pp_dp"],
        "stage_layers": plan["stage_layers"],
        "step_ms": 1000 * step_seconds,
        "forward_backward_ms": 1000 * mean(value for run in runs for value in run["forward_backward_seconds"]),
        "tokens_per_second": 2048 / step_seconds,
        "round_tokens_per_second": [2048 / mean(run["step_seconds"]) for run in runs],
        "verified_optimizer_updates_per_rank_per_run": 7,
    }


def main():
    output = Path(sys.argv[1])
    workload = sys.argv[2:]
    scores = read_json(output / "candidates.json")
    selection = read_json(output / "selection.json")
    plans = {
        score["case"]: score
        for score in scores if score["estimated_forward_backward_ms"] is not None
    }
    labels = [label for label in ("tp2", "pp2_uniform", "pp2_auto") if label in plans]
    failures = {}
    measurements = {label: [] for label in labels}

    def record_failure(label, phase, error):
        failures[label] = {"case": label, "phase": phase, "error": str(error)}
        print("TEST5_ASCEND_4L_FAILURE", json.dumps(failures[label]), flush=True)

    for label in labels:
        plan_file = output / f"{label}-plan.json"
        plan = {**plans[label], "sequence_length": 256}
        plan_file.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"TEST5_ASCEND_4L_START case={label} phase=verify", flush=True)
        try:
            launch(output, "verify", label, 0, plan_file, output / "verify-unused.json", workload)
        except RuntimeError as error:
            record_failure(label, "verify", error)

    for pair, order in enumerate((labels, list(reversed(labels))), 1):
        for label in order:
            if label in failures:
                continue
            print(f"TEST5_ASCEND_4L_START case={label} phase=timed round={pair}", flush=True)
            result_file = output / f"{pair}-{label}.json"
            try:
                launch(output, "timed", label, pair, output / f"{label}-plan.json", result_file, workload)
                result = read_json(result_file)
                check_result(result, label, pair, plans[label])
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                record_failure(label, f"timed round {pair}", error)
                continue
            measurements[label].append(result)

    results = {
        label: summarize(plans[label], runs)
        for label, runs in measurements.items() if len(runs) == 2
    }
    selected = selection["after_vocab"]["case"]
    change = None
    if selected != "tp2" and selected in results and "tp2" in results:
        change = 100 * (results[selected]["tokens_per_second"] / results["tp2"]["tokens_per_second"] - 1)
    manual = "pp2_uniform"
    manual_change = None
    if selected in results and manual in results:
        manual_change = 100 * (results[selected]["tokens_per_second"] / results[manual]["tokens_per_second"] - 1)
    summary = {
        "num_layers": 4,
        "sequence_length": 256,
        "global_batch_size": 8,
        "transport": "native_hccl",
        "profile_source": (output / "profile-source.txt").read_text(encoding="utf-8").strip(),
        "vocab_communication_profile": read_json(output / "vocab-communication.json"),
        "rounds": 2,
        "warmup_steps_per_run": 2,
        "timed_steps_per_run": 5,
        "candidate_scores": scores,
        "arena_selected_before_vocab": selection["before_vocab"],
        "arena_selected": selection["after_vocab"],
        "fastest_measured_layout": max(results, key=lambda label: results[label]["tokens_per_second"]) if results else None,
        "measured_layouts": results,
        "manual_layout": manual,
        "arena_vs_manual_throughput_pct": manual_change,
        "arena_vs_tp2_throughput_pct": change,
        "runtime_failures": list(failures.values()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("TEST5_ASCEND_4L_VOCAB_COMPARISON", json.dumps(summary), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
