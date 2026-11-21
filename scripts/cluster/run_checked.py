"""Run one fresh policy, reject simulator overflows, and validate its outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

TASKS = (
  "Wbc-G1",
  "Wbc-G1-Preview",
  "Wbc-G1-SE",
  "Wbc-G1-Zest",
  "Wbc-G1-Zest-SE",
  "Wbc-G1-EE",
  "Wbc-G1-EE-SE",
)


def check_simulator_output(line):
  if re.search(r"\boverflow\b", line, re.IGNORECASE):
    raise RuntimeError("Simulator capacity overflow; stopping this policy")
  if "[NanGuard] Detected NaN/Inf" in line:
    # NanGuard prints this only after writing the state dump and model.
    raise RuntimeError("Nonfinite physics state; see persistent diagnostic dump")


def run_monitored(command):
  """Run a child with the same fatal physics checks for training and evaluation."""
  process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    start_new_session=True,
  )
  try:
    assert process.stdout is not None
    for line in process.stdout:
      print(line, end="", flush=True)
      check_simulator_output(line)
    return process.wait()
  finally:
    if process.poll() is None:
      os.killpg(process.pid, signal.SIGTERM)
      try:
        process.wait(timeout=30)
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def finite_tensors(value):
  if isinstance(value, torch.Tensor):
    if not torch.isfinite(value).all():
      raise ValueError("Checkpoint contains a nonfinite tensor")
  elif isinstance(value, dict):
    for child in value.values():
      finite_tensors(child)
  elif isinstance(value, (tuple, list)):
    for child in value:
      finite_tensors(child)


def validate_outputs(args, run_name):
  experiment = args.task.lower().replace("-", "_")
  candidates = list((args.run_root / args.phase / experiment).glob(f"*_{run_name}"))
  if len(candidates) != 1:
    raise ValueError(f"Expected one run directory, found {candidates}")
  run = candidates[0]
  checkpoint = run / f"model_{args.iterations - 1}.pt"
  state = torch.load(checkpoint, map_location="cpu", weights_only=False)
  if state["iter"] != args.iterations - 1:
    raise ValueError(f"Unexpected final iteration: {state['iter']}")
  finite_tensors(state)
  for filename in (
    "env.yaml",
    "agent.yaml",
    "policy.onnx",
    "config.yaml",
    "rsi_bin_stats.npz",
  ):
    if not (run / "params" / filename).is_file():
      raise ValueError(f"Missing output: {filename}")
  # BaseLoader reads saved YAML tags without constructing Python objects.
  env = yaml.load((run / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
  agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
  expected = (
    (int(env["scene"]["num_envs"]), args.num_envs),
    (int(env["sim"]["njmax"]), args.njmax),
    (int(env["sim"]["contact_sensor_maxmatch"]), 128),
    (env["sim"]["nan_guard"]["enabled"].lower(), "true"),
    (int(env["sim"]["nan_guard"]["buffer_size"]), round(0.16 / args.physics_dt)),
    (env["sim"]["nan_guard"]["output_dir"], str(diagnostic_path(args))),
    (env["observations"]["actor"]["nan_policy"], "error"),
    (env["observations"]["critic"]["nan_policy"], "error"),
    (int(agent["seed"]), args.seed),
    (int(agent["max_iterations"]), args.iterations),
    (int(agent["num_steps_per_env"]), 24),
    (agent["resume"].lower(), "false"),
    (float(env["sim"]["mujoco"]["timestep"]), args.physics_dt),
    (int(env["decimation"]), round(0.02 / args.physics_dt)),
    (env["scene"]["sensors"][1]["name"], "self_collision"),
    (int(env["scene"]["sensors"][1]["history_length"]), round(0.02 / args.physics_dt)),
    (env["sim"]["mujoco"]["integrator"], "implicitfast"),
    (env["sim"]["mujoco"]["solver"], "newton"),
    (int(env["sim"]["mujoco"]["iterations"]), 10),
    (int(env["sim"]["mujoco"]["ls_iterations"]), 20),
    (env["sim"]["mujoco"]["cone"], "pyramidal"),
    (float(env["sim"]["mujoco"]["impratio"]), 1.0),
    (
      float(env["rewards"]["actuator_torque_soft_limit"]["weight"]),
      -0.1 if args.torque_penalty == "corrected" else 0.0,
    ),
  )
  if any(actual != wanted for actual, wanted in expected):
    raise ValueError(
      f"Saved configuration does not match requested settings: {expected}"
    )
  return str(checkpoint)


def diagnostic_path(args):
  return args.run_root / "diagnostics" / f"{args.phase}_{args.task}_job{args.job_id}"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True, choices=TASKS)
  parser.add_argument("--phase", required=True, choices=("smoke", "stability", "train"))
  parser.add_argument("--iterations", type=int, required=True)
  parser.add_argument("--run-root", type=Path, default=Path("/runs"))
  parser.add_argument("--job-id", required=True)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--dataset-path", default="/workspace/wbc-mjlab/data/g1/samples")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--njmax", type=int, default=1024)
  parser.add_argument(
    "--physics-dt", type=float, choices=(0.005, 0.0025, 0.001), default=0.005
  )
  parser.add_argument(
    "--torque-penalty", choices=("corrected", "disabled"), default="corrected"
  )
  parser.add_argument(
    "--evaluate",
    action="store_true",
    help="Evaluate the final checkpoint before declaring the job complete",
  )
  args = parser.parse_args()
  for count in (args.iterations, args.num_envs, args.njmax):
    if count <= 0:
      parser.error("Counts must be positive")

  run_name = f"{args.phase}_seed{args.seed}_job{args.job_id}"
  command = [
    sys.executable,
    "-m",
    "wbc_mjlab.scripts.train",
    "--task",
    args.task,
    "--dataset-path",
    args.dataset_path,
    "--env.scene.num-envs",
    str(args.num_envs),
    "--env.sim.njmax",
    str(args.njmax),
    "--env.sim.contact-sensor-maxmatch",
    "128",
    "--enable-nan-guard",
    "True",
    "--env.sim.nan-guard.buffer-size",
    str(round(0.16 / args.physics_dt)),
    "--env.sim.nan-guard.output-dir",
    str(diagnostic_path(args)),
    "--env.observations.actor.nan-policy",
    "error",
    "--env.observations.critic.nan-policy",
    "error",
    "--agent.seed",
    str(args.seed),
    "--agent.resume",
    "False",
    "--agent.max-iterations",
    str(args.iterations),
    "--agent.num-steps-per-env",
    "24",
    "--agent.save-interval",
    "5000",
    "--agent.logger",
    "tensorboard",
    "--agent.run-name",
    run_name,
    "--log-root",
    str(args.run_root / args.phase),
    "--env.sim.mujoco.timestep",
    str(args.physics_dt),
    "--env.decimation",
    str(round(0.02 / args.physics_dt)),
    "--env.scene.sensors.1.history-length",
    str(round(0.02 / args.physics_dt)),
    "--env.rewards.actuator-torque-soft-limit.weight",
    "-0.1" if args.torque_penalty == "corrected" else "0.0",
  ]
  start = time.time()
  result = dict(
    task=args.task,
    phase=args.phase,
    job_id=args.job_id,
    node=os.uname().nodename,
    started=start,
    command=command,
    diagnostics=str(diagnostic_path(args)),
    versions={
      name: importlib.metadata.version(name)
      for name in (
        "mjlab",
        "mujoco",
        "mujoco-warp",
        "warp-lang",
        "torch",
      )
    },
    actuator_model="unitree_speed_envelope_then_passive_friction_v1",
    torque_penalty=args.torque_penalty,
  )
  snapshot = Path("SNAPSHOT.sha256")
  if snapshot.is_file():
    result["snapshot_manifest_sha256"] = hashlib.sha256(
      snapshot.read_bytes()
    ).hexdigest()
  status = 1
  try:
    print("COMMAND=" + json.dumps(command), flush=True)
    status = run_monitored(command)
    if status != 0:
      raise RuntimeError(f"Training exited with status {status}")
    result["checkpoint"] = validate_outputs(args, run_name)
    if args.evaluate:
      evaluation = (
        args.run_root / "evaluation" / f"{args.phase}_{args.task}_job{args.job_id}"
      )
      evaluation_command = [
        sys.executable,
        "-m",
        "wbc_mjlab.scripts.evaluate",
        "--task",
        args.task,
        "--checkpoint",
        result["checkpoint"],
        "--onnx-file",
        str(Path(result["checkpoint"]).parent / "params/policy.onnx"),
        "--motion-file",
        args.dataset_path,
        "--physics-dt",
        str(args.physics_dt),
        "--output",
        str(evaluation),
      ]
      result["evaluation_command"] = evaluation_command
      print("EVALUATION=" + json.dumps(evaluation_command), flush=True)
      status = run_monitored(evaluation_command)
      if status != 0:
        raise RuntimeError(f"Evaluation exited with status {status}")
      result["evaluation"] = str(evaluation / "results.json")
    result["state"] = "completed"
    status = 0
  except Exception as error:
    print(f"JOB FAILED: {error}", file=sys.stderr, flush=True)
    result.update(state="failed", error=str(error))
    status = status or 1
  finally:
    result.update(
      exit_code=status, finished=time.time(), elapsed_seconds=time.time() - start
    )
    output = (
      args.run_root / "status" / f"{args.phase}_{args.task}_job{args.job_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
  raise SystemExit(status)


if __name__ == "__main__":
  main()
