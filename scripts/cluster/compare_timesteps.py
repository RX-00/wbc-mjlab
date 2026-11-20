"""Evaluate three fixed pilot policies at 5, 2.5 and 1 ms with 50 Hz control."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from run_checked import TASKS as ALL_TASKS
from run_checked import run_monitored

TASKS = ("Wbc-G1-SE", "Wbc-G1-Zest", "Wbc-G1")
TIMESTEPS = (0.005, 0.0025, 0.001)


def compare_traces(candidate_dir, reference_dir):
  candidate = json.loads((candidate_dir / "results.json").read_text())
  reference = json.loads((reference_dir / "results.json").read_text())
  with (
    np.load(candidate_dir / "initial_states.npz") as a,
    np.load(reference_dir / "initial_states.npz") as b,
  ):
    np.testing.assert_array_equal(a["state"], b["state"])
    np.testing.assert_array_equal(a["ctrl"], b["ctrl"])
  differences = []
  for left, right in zip(candidate["trials"], reference["trials"], strict=True):
    assert (left["clip"], left["start"], left["steps"]) == (
      right["clip"],
      right["start"],
      right["steps"],
    )
    with (
      np.load(candidate_dir / left["trace"]) as a,
      np.load(reference_dir / right["trace"]) as b,
    ):
      count = min(50, len(a["qpos"]), len(b["qpos"]))
      if not count:
        differences.append(
          {"clip": left["name"], "suite": left["suite"], "compared_steps": 0}
        )
        continue
      np.testing.assert_array_equal(a["frame"][:count], b["frame"][:count])
      qa, qb = a["qpos"][:count], b["qpos"][:count]
      differences.append(
        {
          "clip": left["name"],
          "suite": left["suite"],
          "compared_steps": count,
          "joint_position_rmse_rad": float(
            np.sqrt(np.mean((qa[:, 7:] - qb[:, 7:]) ** 2))
          ),
          "base_position_mean_m": float(
            np.linalg.norm(qa[:, :3] - qb[:, :3], axis=-1).mean()
          ),
          "joint_velocity_rmse_rad_s": float(
            np.sqrt(np.mean((a["qvel"][:count, 6:] - b["qvel"][:count, 6:]) ** 2))
          ),
        }
      )
  return {
    "reference_dt": 0.001,
    "scope": "First second of shared coverage per trial; fine step is a sensitivity reference, not ground truth",
    "trials": differences,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--pilot-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--phase", choices=["stability", "train"], default="stability")
  parser.add_argument(
    "--actuator-model",
    choices=["corrected", "historical_unbounded"],
    default="corrected",
  )
  parser.add_argument("--tasks", nargs="+", choices=ALL_TASKS, default=TASKS)
  parser.add_argument(
    "--timesteps", nargs="+", type=float, choices=TIMESTEPS, default=TIMESTEPS
  )
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  result = {
    "control_period_s": 0.02,
    "checkpoint_phase": args.phase,
    "actuator_model": args.actuator_model,
    "evaluations": [],
    "state": "running",
  }
  try:
    for task in args.tasks:
      matches = list(
        (args.pilot_root / "status").glob(f"{args.phase}_{task}_job*.json")
      )
      if len(matches) != 1:
        raise ValueError(f"Expected one completed pilot status for {task}: {matches}")
      status = json.loads(matches[0].read_text())
      if status["state"] != "completed":
        raise ValueError(f"Pilot is not complete: {matches[0]}")
      # Training mounted its run root at /runs; this job mounts the parent root.
      checkpoint = args.pilot_root / Path(status["checkpoint"]).relative_to("/runs")
      for dt in args.timesteps:
        output = args.output / f"{task}_{dt}"
        command = [
          sys.executable,
          "-m",
          "wbc_mjlab.scripts.evaluate",
          "--task",
          task,
          "--checkpoint",
          str(checkpoint),
          "--onnx-file",
          str(checkpoint.parent / "params/policy.onnx"),
          "--motion-file",
          "/workspace/wbc-mjlab/data/g1/samples",
          "--output",
          str(output),
          "--physics-dt",
          str(dt),
          "--actuator-model",
          args.actuator_model,
        ]
        record = {
          "task": task,
          "physics_dt": dt,
          "command": command,
          "state": "running",
        }
        result["evaluations"].append(record)
        (args.output / "comparison.json").write_text(
          json.dumps(result, indent=2) + "\n"
        )
        print("EVALUATION=" + json.dumps(command), flush=True)
        code = run_monitored(command)
        if code:
          record.update(state="failed", exit_code=code)
          raise RuntimeError(f"Evaluation failed for {task} at {dt}: {code}")
        evaluation = json.loads((output / "results.json").read_text())
        seconds = sum(trial["observed_seconds"] for trial in evaluation["trials"])
        wall_time = sum(trial["loop_wall_seconds"] for trial in evaluation["trials"])
        record.update(
          state="completed",
          results=str(output / "results.json"),
          summary=evaluation["summary"],
          simulated_seconds=seconds,
          loop_wall_seconds=wall_time,
          simulated_seconds_per_wall_second=seconds / wall_time,
          selection="Pending review of stability, tracking, contacts, short-trace error and runtime",
        )
      for record in result["evaluations"]:
        if (
          0.001 in args.timesteps
          and record["task"] == task
          and record["physics_dt"] != 0.001
        ):
          record["short_trace_comparison"] = compare_traces(
            args.output / f"{task}_{record['physics_dt']}",
            args.output / f"{task}_0.001",
          )
    result["state"] = "completed"
  except Exception as error:
    result.update(state="failed", error=str(error))
    raise
  finally:
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
  main()
