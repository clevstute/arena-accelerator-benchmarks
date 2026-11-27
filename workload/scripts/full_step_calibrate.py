"""Record each rank's training-step boundaries during a calibration replay."""

import json
from socket import gethostname
from time import monotonic_ns
from unittest.mock import patch

import replay


class CalibrationTrainer(replay.TimedTrainer):
    def train(self, metric_types=()):
        outer_ticks = []
        forward_starts = []
        backward_ends = []

        def step_clock():
            timestamp = monotonic_ns()
            outer_ticks.append(timestamp)
            return timestamp / 1_000_000_000

        def observe_timer(call, timestamps):
            def wrapped(opcode, *args, **kwargs):
                result = call(opcode, *args, **kwargs)
                if opcode == "train_step":
                    timestamps.append(monotonic_ns())
                return result

            return wrapped

        with (
            patch.object(replay, "perf_counter", step_clock),
            patch.object(self.timer, "start", observe_timer(self.timer.start, forward_starts)),
            patch.object(self.timer, "stop", observe_timer(self.timer.stop, backward_ends)),
        ):
            result = super().train(metric_types)

        steps = self.run.steps
        if len(outer_ticks) != 2 * steps or len(forward_starts) != steps or len(backward_ends) != steps:
            raise RuntimeError(f"Incomplete timing boundaries for {self.run.label}.")
        boundaries = list(zip(outer_ticks[::2], forward_starts, backward_ends, outer_ticks[1::2], strict=True))
        rank = replay.dist.get_rank()
        path = self.run.output.with_name(f"{self.run.output.stem}.rank{rank}.json")
        payload = {
            "case": self.run.label,
            "rank": rank,
            "host": gethostname(),
            "steps_ns": boundaries[replay.WARMUP:],
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return result


if __name__ == "__main__":
    replay.TimedTrainer = CalibrationTrainer
    replay.main()
