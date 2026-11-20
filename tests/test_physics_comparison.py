"""Matched trajectories must compare the same physical starts and frames."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_comparison(monkeypatch):
  scripts = Path(__file__).parents[1] / "scripts/cluster"
  monkeypatch.syspath_prepend(str(scripts))
  spec = importlib.util.spec_from_file_location(
    "compare_timesteps", scripts / "compare_timesteps.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_comparison_checks_initial_state_and_reports_common_coverage(
  tmp_path, monkeypatch
):
  module = load_comparison(monkeypatch)
  reference, candidate = tmp_path / "reference", tmp_path / "candidate"
  for folder, count in ((reference, 3), (candidate, 2)):
    folder.mkdir()
    np.savez(
      folder / "initial_states.npz", state=np.zeros((1, 15)), ctrl=np.zeros((1, 1))
    )
    qpos, qvel = np.zeros((count, 8)), np.zeros((count, 7))
    if folder == candidate:
      qpos[:, 0], qpos[:, 7], qvel[:, 6] = 0.2, 0.1, 0.3
    np.savez(folder / "trace.npz", qpos=qpos, qvel=qvel, frame=np.arange(count))
    (folder / "results.json").write_text(
      json.dumps(
        {
          "trials": [
            {
              "clip": 0,
              "name": "clip",
              "start": 0,
              "steps": 3,
              "suite": "full",
              "trace": "trace.npz",
            }
          ]
        }
      )
    )
  trial = module.compare_traces(candidate, reference)["trials"][0]
  assert trial["compared_steps"] == 2
  assert trial["base_position_mean_m"] == pytest.approx(0.2)
  assert trial["joint_position_rmse_rad"] == pytest.approx(0.1)
  assert trial["joint_velocity_rmse_rad_s"] == pytest.approx(0.3)
  np.savez(
    candidate / "initial_states.npz", state=np.ones((1, 15)), ctrl=np.zeros((1, 1))
  )
  with pytest.raises(AssertionError):
    module.compare_traces(candidate, reference)


def test_historical_batch_selects_final_checkpoint_and_matching_export(
  tmp_path, monkeypatch
):
  import sys

  module = load_comparison(monkeypatch)
  root, output = tmp_path / "campaign", tmp_path / "evaluation"
  (root / "status").mkdir(parents=True)
  (root / "status/train_Wbc-G1_job1.json").write_text(
    json.dumps(
      {
        "state": "completed",
        "checkpoint": "/runs/train/g1/model_199999.pt",
      }
    )
  )
  # A same-task pilot must not be substituted for the requested final checkpoint.
  (root / "status/stability_Wbc-G1_job2.json").write_text(
    json.dumps(
      {
        "state": "failed",
        "checkpoint": "/runs/stability/g1/model_999.pt",
      }
    )
  )
  commands = []

  def evaluate(command):
    commands.append(command)
    folder = Path(command[command.index("--output") + 1])
    folder.mkdir()
    (folder / "results.json").write_text(
      json.dumps(
        {
          "summary": {},
          "trials": [{"observed_seconds": 1.0, "loop_wall_seconds": 2.0}],
        }
      )
    )
    return 0

  monkeypatch.setattr(module, "run_monitored", evaluate)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "compare_timesteps",
      "--pilot-root",
      str(root),
      "--output",
      str(output),
      "--phase",
      "train",
      "--tasks",
      "Wbc-G1",
      "--timesteps",
      "0.005",
      "--actuator-model",
      "historical_unbounded",
    ],
  )
  module.main()
  assert len(commands) == 1
  command = commands[0]
  assert command[command.index("--checkpoint") + 1] == str(
    root / "train/g1/model_199999.pt"
  )
  assert command[command.index("--onnx-file") + 1] == str(
    root / "train/g1/params/policy.onnx"
  )
  assert command[command.index("--actuator-model") + 1] == "historical_unbounded"
  result = json.loads((output / "comparison.json").read_text())
  assert result["state"] == "completed"
  assert result["checkpoint_phase"] == "train"
  assert result["actuator_model"] == "historical_unbounded"
