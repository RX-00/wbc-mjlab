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
    "broadphase overflow - please increase nconmax to 52 or naconmax to 52",
    "narrowphase overflow - please increase nconmax",
    "njmax_nnz overflow",
    "EPA horizon buffer overflow",
    "contact match overflow: please increase Option.contact_sensor_maxmatch to 65",
    "[NanGuard] Detected NaN/Inf at step 1234",
  ],
)
def test_simulator_failure_is_fatal(line):
  with pytest.raises(RuntimeError):
    checks.check_simulator_output(line)


def test_normal_output_is_accepted():
  checks.check_simulator_output("Learning iteration 123/1000")


def test_six_policy_submission_orders_dependencies_and_preserves_profile(tmp_path):
  import json
  import os
  import subprocess

  binary = tmp_path / "bin"
  binary.mkdir()
  ledger = tmp_path / "submissions.json"
  sbatch = binary / "sbatch"
  sbatch.write_text(
    "#!/usr/bin/env python3\n"
    "import json, os, sys\n"
    "from pathlib import Path\n"
    "p = Path(os.environ['TEST_SUBMISSIONS'])\n"
    "rows = json.loads(p.read_text()) if p.exists() else []\n"
    "rows.append(sys.argv[1:])\n"
    "p.write_text(json.dumps(rows))\n"
    "print(100 + len(rows))\n"
  )
  sbatch.chmod(0o755)
  image, root = tmp_path / "image.sif", tmp_path / "run with spaces"
  image.touch()
  script = Path(__file__).parents[1] / "scripts/cluster/submit_six.sh"
  command = [
    "bash",
    str(script),
    str(image),
    str(root),
    "train",
    "--physics-dt",
    "0.0025",
    "--evaluate",
  ]
  environment = {
    **os.environ,
    "PATH": str(binary) + os.pathsep + os.environ["PATH"],
    "TEST_SUBMISSIONS": str(ledger),
  }
  subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
  rows = json.loads(ledger.read_text())
  tasks = [
    "Wbc-G1-SE",
    "Wbc-G1-Zest",
    "Wbc-G1",
    "Wbc-G1-Zest-SE",
    "Wbc-G1-EE",
    "Wbc-G1-EE-SE",
  ]
  assert len(rows) == 6
  for index, (row, task) in enumerate(zip(rows, tasks, strict=True)):
    assert f"--job-name=wbc_train_{task}" in row
    assert row[-3:] == ["--physics-dt", "0.0025", "--evaluate"]
    position = row.index(task)
    assert row[position + 1 : position + 3] == ["train", "200000"]
    assert row[position + 4 : position + 6] == [str(root), str(image)]
    dependencies = [item for item in row if item.startswith("--dependency=")]
    assert dependencies == (
      [] if index == 0 else [f"--dependency=afterok:{100 + index}"]
    )
  # A repeated invocation must not create a duplicate campaign.
  result = subprocess.run(command, env=environment, capture_output=True, text=True)
  assert result.returncode != 0
  assert json.loads(ledger.read_text()) == rows


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_optimizer_state_rejected(bad):
  with pytest.raises(ValueError, match="nonfinite"):
    checks.finite_tensors({"optimizer": {"state": [torch.tensor([1.0, bad])]}})


@pytest.mark.parametrize(
  "line",
  [
    "contact match overflow: please increase Option.contact_sensor_maxmatch to 65",
    "broadphase overflow - please increase nconmax to 52 or naconmax to 52",
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
