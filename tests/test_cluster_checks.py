"""Failures seen in campaign logs must fail the cluster job."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
  "run_checked", Path(__file__).parents[1] / "scripts/cluster/run_checked.py"
)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.mark.parametrize(
  "line",
  [
    "WARNING: nefc overflow: please increase njmax",
    "WARNING: ncon overflow",
    "contact match overflow: please increase Option.contact_sensor_maxmatch to 65",
    "[NanGuard] Detected NaN/Inf at step 1234",
  ],
)
def test_simulator_failure_is_fatal(line):
  with pytest.raises(RuntimeError):
    checks.check_simulator_output(line)


def test_normal_output_is_accepted():
  checks.check_simulator_output("Learning iteration 123/1000")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_optimizer_state_rejected(bad):
  with pytest.raises(ValueError, match="nonfinite"):
    checks.finite_tensors({"optimizer": {"state": [torch.tensor([1.0, bad])]}})


@pytest.mark.parametrize(
  "line",
  [
    "contact match overflow: please increase Option.contact_sensor_maxmatch to 65",
    "[NanGuard] Detected NaN/Inf at step 1234",
  ],
)
def test_failure_stops_child_and_writes_failed_status(monkeypatch, tmp_path, line):
  import json
  import subprocess
  import sys

  popen = subprocess.Popen
  children = []

  def failing_training(command, **kwargs):
    child = popen(
      [
        sys.executable,
        "-u",
        "-c",
        f"import time; print({line!r}, flush=True); time.sleep(60)",
      ],
      **kwargs,
    )
    children.append(child)
    return child

  monkeypatch.setattr(checks.subprocess, "Popen", failing_training)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "run_checked.py",
      "--task",
      "Wbc-G1",
      "--phase",
      "smoke",
      "--iterations",
      "10",
      "--job-id",
      "test",
      "--run-root",
      str(tmp_path),
    ],
  )
  with pytest.raises(SystemExit) as error:
    checks.main()
  assert error.value.code != 0
  assert children[0].poll() is not None
  result = json.loads((tmp_path / "status/smoke_Wbc-G1_jobtest.json").read_text())
  assert result["state"] == "failed"
  assert result["exit_code"] != 0
  assert "checkpoint" not in result
