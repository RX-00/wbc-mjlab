"""Continue a failed matched PPO allocation using its native model and optimizer state."""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

# Executed in the original sealed training container via scripts/cluster/check.sbatch.
sys.path.insert(0, str(Path("scripts/cluster").resolve()))
from run_checked import finite_tensors, run_monitored


def digest(path):
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("original_status", type=Path)
  p.add_argument("checkpoint", type=Path)
  p.add_argument("sha256")
  args = p.parse_args()
  old = json.loads(args.original_status.read_text())
  command = old["command"].copy()
  if old["state"] != "failed" or digest(args.checkpoint) != args.sha256:
    raise ValueError("Unexpected original state or recovery checkpoint identity")
  frozen = digest(Path("SNAPSHOT.sha256"))
  if frozen != old["snapshot_manifest_sha256"]:
    raise ValueError("Training source differs from the original allocation")
  job = os.environ["COMPARISON_JOB_ID"]
  name = f"train_seed44_job{job}"
  status_file = args.original_status.with_name(f"train_Wbc-G1-Preview_job{job}.json")
  row = dict(
    job=job,
    previous_job=old["job_id"],
    source_manifest=frozen,
    parent=str(args.checkpoint),
    parent_sha256=args.sha256,
    started=time.time(),
    state="running",
  )
  try:
    state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    finite_tensors(state)
    parent = state["iter"]
    total = int(command[command.index("--agent.max-iterations") + 1])
    if parent != 100000 or total != 200000 or not any("optimizer" in k for k in state):
      raise ValueError("Unexpected training cursor or missing optimizer state")
    # RSL-RL saves after update i and starts the next learn loop at the loaded
    # index. Advance the cursor in a separate input, preserving the original.
    resume_dir = args.checkpoint.parent.parent / f"resume_input_job{job}"
    resume_dir.mkdir()
    state["iter"] = parent + 1
    input_file = resume_dir / "model_resume.pt"
    torch.save(state, input_file)
    del state
    (resume_dir / "params").symlink_to(
      args.checkpoint.parent / "params", target_is_directory=True
    )
    for key, value in {
      "--agent.resume": "True",
      "--agent.run-name": name,
      "--agent.max-iterations": str(total - parent - 1),
      "--env.sim.nan-guard.output-dir": f"/runs/diagnostics/train_Wbc-G1-Preview_job{job}",
    }.items():
      command[command.index(key) + 1] = value
    command.extend(
      ["--agent.load-run", resume_dir.name, "--agent.load-checkpoint", input_file.name]
    )
    row.update(
      next_iteration=parent + 1, command=command, input_sha256=digest(input_file)
    )
    status_file.write_text(json.dumps(row, indent=2) + "\n")
    print("RECOVERY", json.dumps(row), flush=True)
    result = run_monitored(command)
    if result:
      raise RuntimeError(f"Resumed training exited with status {result}")
    (run,) = args.checkpoint.parent.parent.glob(f"*_{name}")
    final = run / f"model_{total - 1}.pt"
    state = torch.load(final, weights_only=False, map_location="cpu")
    finite_tensors(state)
    if state["iter"] != total - 1:
      raise ValueError("Unexpected final iteration")
    for filename in (
      "env.yaml",
      "agent.yaml",
      "policy.onnx",
      "config.yaml",
      "rsi_bin_stats.npz",
    ):
      if not (run / "params" / filename).is_file():
        raise FileNotFoundError(filename)
    agent = yaml.load((run / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
    if agent["resume"].lower() != "true" or int(agent["seed"]) != 44:
      raise ValueError("Incorrect resumed configuration")
    row.update(
      state="completed", checkpoint=str(final), checkpoint_sha256=digest(final)
    )
  except BaseException as exc:
    row.update(state="failed", error=str(exc))
    raise
  finally:
    row["finished"] = time.time()
    status_file.write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
  main()
