"""Evaluation schedule, denominators and nominal configuration."""

from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch

from wbc_mjlab.robots.g1.tasks import make_g1_wbc_env_cfg
from wbc_mjlab.scripts.evaluate import (
  configure_evaluation,
  load_onnx_policy,
  summarize,
  trial_schedule,
)


@pytest.mark.parametrize("alpha", [0.0, 0.2, 1.0])
def test_reference_elu_large_positive_branch_matches_torch(tmp_path, alpha):
  node = onnx.helper.make_node("Elu", ["x"], ["y"], alpha=alpha)

  def info(name):
    return onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [9])

  model = onnx.helper.make_model(
    onnx.helper.make_graph([node], "elu", [info("x")], [info("y")]),
    opset_imports=[onnx.helper.make_opsetid("", 17)],
  )
  path = tmp_path / "elu.onnx"
  onnx.save(model, path)
  policy = load_onnx_policy(path)
  values = np.array([-1000, -100, -1, -1e-4, 0, 1e-4, 1, 100, 1000], np.float32)
  # The upstream eager where implementation raises here on exp(100/1000).
  with np.errstate(over="raise", invalid="raise"):
    actual = policy.run(None, {"x": values})[0]
  expected = torch.nn.functional.elu(torch.from_numpy(values), alpha).numpy()
  np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
  assert actual.dtype == values.dtype
  assert np.array_equal(actual[values >= 0], values[values >= 0])
  # Invalid upstream activations must remain visible to the parity/finiteness gate.
  values[-2:] = [np.inf, np.nan]
  actual = policy.run(None, {"x": values})[0]
  assert np.isposinf(actual[-2]) and np.isnan(actual[-1])


def test_schedule_never_steps_past_last_frame():
  motion = SimpleNamespace(
    segment_names=("short", "long"),
    segment_start_idx=torch.tensor([0, 10]),
    segment_length=torch.tensor([10, 1000]),
  )
  trials = trial_schedule(motion)
  assert len(trials) == 8
  assert [t["steps"] for t in trials if t["suite"] == "full"] == [9, 999]
  for trial in trials:
    end = (10, 1010)[trial["clip"]]
    assert trial["start"] + trial["steps"] <= end - 1


def test_summary_keeps_falls_and_unstarted_trials_in_denominator():
  result = summarize(
    [
      dict(suite="full", status="completed", steps=10, observed_steps=10),
      dict(suite="full", status="fall", steps=10, observed_steps=2),
      dict(suite="full", status="numerical_failure", steps=10, observed_steps=1),
      dict(suite="full", status="not_started", steps=10),
    ]
  )["full"]
  assert result["completion_rate"] == 0.25
  assert result["observed_steps"] == 13
  assert result["falls"] == result["numerical_failures"] == 1


def test_nominal_evaluation_disables_all_randomization_and_assistance(tmp_path):
  cfg = make_g1_wbc_env_cfg(play=True, task_id="Wbc-G1-SE")
  configure_evaluation(cfg, tmp_path, tmp_path, 0.0025)
  assert cfg.events == {}
  assert cfg.terminations == {}
  assert not cfg.auto_reset
  assert cfg.decimation == 8
  assert cfg.scene.sensors[1].history_length == 8
  assert cfg.sim.nan_guard.buffer_size == 64
  assert cfg.sim.nconmax == 128
  motion = cfg.commands["motion"]
  assert not motion.assistive_wrench_enabled
  assert motion.joint_position_range == (0, 0)
  assert not motion.pose_range and not motion.velocity_range
  assert motion.rsi.sampling_mode == "start"
