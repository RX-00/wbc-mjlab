"""Exercise reset and clip boundaries with real simulator state."""

import os

os.environ.setdefault("MUJOCO_GL", "disable")

import mujoco
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.nan_guard import NanGuard, NanGuardCfg

from wbc_mjlab.robots.g1.constants import get_spec
from wbc_mjlab.robots.g1.tasks import get_g1_task_config


@pytest.fixture(scope="module")
def env(tmp_path_factory):
  model = get_spec().compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  frames = 64
  path = tmp_path_factory.mktemp("motion") / "standing.npz"
  np.savez(
    path,
    fps=np.array([50.0]),
    joint_pos=np.tile(data.qpos[7:], (frames, 1)),
    joint_vel=np.zeros((frames, model.nv - 6)),
    body_pos_w=np.tile(data.xpos[None, 1:], (frames, 1, 1)),
    body_quat_w=np.tile(data.xquat[None, 1:], (frames, 1, 1)),
    body_lin_vel_w=np.zeros((frames, model.nbody - 1, 3)),
    body_ang_vel_w=np.zeros((frames, model.nbody - 1, 3)),
    segment_start_idx=np.array([0, frames // 2]),
    segment_length=np.array([frames // 2, frames // 2]),
  )
  cfg = get_g1_task_config("Wbc-G1-SE").build_env_cfg()
  cfg.scene.num_envs = 2
  cfg.sim.njmax = 1024
  cfg.seed = 42
  motion = cfg.commands["motion"]
  motion.motion_file = str(path)
  motion.rsi.sampling_mode = "start"
  motion.rsi.bin_width_s = 0.1
  motion.pose_range = {}
  motion.velocity_range = {}
  motion.joint_position_range = (0.0, 0.0)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  instance = ManagerBasedRlEnv(cfg=cfg, device=device)
  yield instance
  instance.close()


@pytest.mark.slow
@pytest.mark.parametrize("reset_kind", ["environment", "viewer", "clip_wrap"])
def test_reset_clears_only_selected_wrench_cache(env, reset_kind):
  with torch.inference_mode():
    env.reset()
    motion = env.command_manager.get_term("motion")
    ids = torch.tensor([0], device=env.device)
    motion.assist_force_w[:] = 17.0
    motion.assist_torque_w[:] = 23.0
    motion.assist_force_w[0] = float("nan")
    motion.assist_torque_w[0] = float("inf")
    if reset_kind == "environment":
      obs, _ = env.reset(env_ids=ids)
      assert all(torch.isfinite(value[0]).all() for value in obs.values())
    elif reset_kind == "viewer":
      motion.reset_to_frame(ids, 0)
    else:
      motion.time_steps[0] = motion.motion.segment_end_idx[0] - 1
      motion.compute(dt=env.step_dt)
    assert torch.equal(motion.assist_force_w[0], torch.zeros(3, device=env.device))
    assert torch.equal(motion.assist_torque_w[0], torch.zeros(3, device=env.device))
    assert (motion.assist_force_w[1] == 17.0).all()
    assert (motion.assist_torque_w[1] == 23.0).all()


@pytest.mark.slow
def test_clip_wrap_refreshes_derived_kinematics(env):
  with torch.inference_mode():
    env.reset()
    motion = env.command_manager.get_term("motion")
    ids = torch.tensor([0], device=env.device)
    data = motion.robot.data
    state = torch.cat(
      [
        data.root_link_pos_w[ids],
        data.root_link_quat_w[ids],
        data.root_link_lin_vel_w[ids],
        data.root_link_ang_vel_w[ids],
      ],
      dim=-1,
    )
    state[:, 0] += 5.0
    motion.robot.write_root_state_to_sim(state, env_ids=ids)
    env.sim.forward()
    motion.time_steps[0] = motion.motion.segment_end_idx[0] - 1
    motion.compute(dt=env.step_dt)
    observed = motion.robot_body_pos_w.clone()
    env.sim.forward()
    torch.testing.assert_close(observed, motion.robot_body_pos_w)


@pytest.mark.slow
def test_reset_to_frame_at_segment_boundary_selects_next_clip(env):
  with torch.inference_mode():
    env.reset()
    motion = env.command_manager.get_term("motion")
    ids = torch.tensor([0], device=env.device)
    start = int(motion.motion.segment_start_idx[1])
    motion.reset_to_frame(ids, start)
    assert motion.trajectory_ids[0] == 1
    assert motion.time_steps[0] == start


@pytest.mark.slow
def test_nonfinite_physics_writes_replayable_dump(env, tmp_path):
  guard = NanGuard(
    NanGuardCfg(enabled=True, output_dir=str(tmp_path), buffer_size=2),
    env.num_envs,
    env.sim.mj_model,
  )
  try:
    with torch.inference_mode():
      env.reset()

      guard.capture(env.sim.data)
      env.sim.data.qvel[0, 0] = float("nan")
      assert guard.check_and_dump(env.sim.data)
    dumps = sorted({p.resolve() for p in tmp_path.glob("nan_dump_*.npz")})
    models = sorted({p.resolve() for p in tmp_path.glob("model_*.mjb")})
    assert len(dumps) == len(models) == 1
    assert (tmp_path / "nan_dump_latest.npz").resolve() == dumps[0]
    assert (tmp_path / "model_latest.mjb").resolve() == models[0]
    with np.load(dumps[0]) as saved:
      assert saved.files
    assert mujoco.MjModel.from_binary_path(str(models[0])).nv == env.sim.mj_model.nv
  finally:
    with torch.inference_mode():
      env.reset()


@pytest.mark.slow
def test_unitree_initialized_controls_and_simulator_effort(env):
  """Real G1 initialization, custom dispatch, ctrl writes and MuJoCo forces."""
  import math

  from mjlab.actuator.fused_group import FusedActuatorGroup

  with torch.inference_mode():
    env.reset()

    robot = env.scene["robot"]
    _, custom = FusedActuatorGroup.process(robot.actuators)
    assert tuple(robot.actuators) == custom
    for speed in (0.0, 10.0, 50.0):
      velocity = torch.full_like(robot.data.joint_vel, speed)
      velocity[1] *= -1
      position = robot.data.joint_pos.clone()
      robot.write_joint_state_to_sim(position, velocity)
      for sign in (-1, 1):
        target = position + sign * 1000
        robot.set_joint_position_target(target)
        robot.set_joint_velocity_target(torch.zeros_like(velocity))
        robot.set_joint_effort_target(torch.zeros_like(velocity))
        env.scene.write_data_to_sim()
        env.sim.forward()
        for actuator in robot.actuators:
          cfg = actuator.cfg
          expected = torch.empty_like(actuator.motor_effort)
          for world, v in enumerate((speed, -speed)):
            peak = cfg.y1 if v * sign > 0 else cfg.y2
            taper = (
              1 if abs(v) <= cfg.x1 else max(0, (cfg.x2 - abs(v)) / (cfg.x2 - cfg.x1))
            )
            friction = cfg.fs * math.tanh(v / cfg.va) + cfg.fd * v
            expected[world] = sign * peak * taper - friction
          torch.testing.assert_close(
            env.sim.data.ctrl[:, actuator.global_ctrl_ids], expected
          )
          torch.testing.assert_close(
            robot.data.actuator_force[:, actuator.ctrl_ids], expected
          )
    env.reset()


@pytest.mark.slow
@pytest.mark.parametrize(
  "stage,field",
  [
    ("pre_step", "ctrl"),
    ("post_step", "qacc"),
    ("sense", "sensordata"),
  ],
)
def test_physical_failure_artifact_precedes_fatal_notification(
  env, tmp_path, stage, field
):
  import json

  from wbc_mjlab.diagnostics import PhysicalFailureCapture

  original_cfg = env.cfg.sim.nan_guard
  env.cfg.sim.nan_guard = NanGuardCfg(
    enabled=True, output_dir=str(tmp_path), buffer_size=2
  )
  guard = PhysicalFailureCapture(env)
  env.cfg.sim.nan_guard = original_cfg
  original_dump = guard._dump_buffer

  def checked_dump(ids):
    assert len(list(tmp_path.glob("physical_failure_*.json"))) == 1
    assert len(list(tmp_path.glob("physical_failure_*.pt"))) == 1
    original_dump(ids)

  guard._dump_buffer = checked_dump
  try:
    with torch.inference_mode():
      env.reset()
      for _ in range(3):
        guard.capture(env.sim.data)
      assert len(guard.buffer) == 2
      tensor = getattr(env.sim.data, field)
      tensor[1, 0] = float("nan")
      with pytest.raises(FloatingPointError, match=stage):
        guard.check(stage, {field: tensor})
    metadata = json.loads(next(tmp_path.glob("physical_failure_*.json")).read_text())
    assert metadata["stage"] == stage
    assert metadata["environment_ids"] == [1]
    assert metadata["faults"][0]["indices"] == [[1, 0]]
    payload = torch.load(
      next(tmp_path.glob("physical_failure_*.pt")), weights_only=True
    )
    assert len(payload["history"]) == 2
    assert "ctrl" in payload["history"][0]
    assert "xfrc_applied" in payload["history"][0]
    assert "body_mass" in payload["model"]
    assert payload["context"]["raw_action"].shape[0] == 1
    assert torch.isnan(payload["detected"][field][0, 0])
  finally:
    with torch.inference_mode():
      env.reset()


@pytest.mark.slow
@pytest.mark.parametrize(
  "physics_dt,actuator_model",
  [
    (0.005, "corrected"),
    (0.0025, "corrected"),
    (0.001, "corrected"),
    (0.005, "historical_unbounded"),
  ],
)
@torch.inference_mode()
def test_headless_evaluator_records_failures_and_export_parity(
  env, tmp_path, monkeypatch, physics_dt, actuator_model
):
  import json
  import sys
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  from wbc_mjlab.rl.runner import PolicyOnlyMotionTrackingRunner
  from wbc_mjlab.scripts.evaluate import main
  from wbc_mjlab.tasks import prepare_wbc_run

  prepare_wbc_run(task_id="Wbc-G1-SE")
  checkpoint = tmp_path / "random.pt"
  agent = load_rl_cfg("Wbc-G1-SE")
  runner = PolicyOnlyMotionTrackingRunner(
    RslRlVecEnvWrapper(env), asdict(agent), device=env.device
  )
  runner.save(str(checkpoint))
  results = []
  for backend in ("torch", "onnx"):
    output = tmp_path / backend
    monkeypatch.setattr(
      sys,
      "argv",
      [
        "evaluate",
        "--task",
        "Wbc-G1-SE",
        "--checkpoint",
        str(checkpoint),
        "--onnx-file",
        str(checkpoint.parent / "params/policy.onnx"),
        "--motion-file",
        env.cfg.commands["motion"].motion_file,
        "--output",
        str(output),
        "--device",
        env.device,
        "--backend",
        backend,
        "--physics-dt",
        str(physics_dt),
        "--actuator-model",
        actuator_model,
      ],
    )
    main()
    result = json.loads((output / "results.json").read_text())
    assert len(result["trials"]) == 8
    assert all(t["status"] in ("completed", "fall") for t in result["trials"])
    assert result["export_parity"]["passed"]
    assert result["export_parity"]["existing_export_checked"]
    assert result["export_parity"]["samples"] >= 8
    for trial in result["trials"]:
      physical = trial["physical"]
      assert (
        physical["motor_joint_samples"]
        == trial["observed_steps"] * round(0.02 / physics_dt) * 29
      )
      assert physical["friction_dissipation_j"] >= 0
      assert 0 <= physical["motor_saturation_fraction"] <= 1
      with np.load(output / trial["trace"]) as trace:
        assert len(trace["qpos"]) == trial["observed_steps"]
    results.append(result)
  for a, b in zip(results[0]["trials"], results[1]["trials"], strict=True):
    assert a["status"] == b["status"]
    assert a["observed_steps"] == b["observed_steps"]
    np.testing.assert_allclose(
      a["terminal_qpos"], b["terminal_qpos"], atol=1e-3, rtol=1e-3
    )
  if physics_dt == 0.005 and actuator_model == "corrected":
    import subprocess

    subprocess.run(
      [
        sys.executable,
        "-m",
        "wbc_mjlab.scripts.render_evaluation",
        "--evaluation",
        str(tmp_path / "torch"),
        "--output",
        str(tmp_path / "video"),
      ],
      env={**os.environ, "MUJOCO_GL": "egl"},
      check=True,
    )
    manifest = json.loads((tmp_path / "video/videos.json").read_text())
    assert manifest["state"] == "completed"
    assert len(manifest["videos"]) == 2
    assert all(
      (tmp_path / "video" / row["video"]).stat().st_size > 1000
      for row in manifest["videos"]
    )


@pytest.mark.slow
@torch.inference_mode()
def test_evaluator_rejects_broadphase_capacity_overflow(env, tmp_path, monkeypatch):
  import json

  from wbc_mjlab.scripts.evaluate import check_collision_capacity

  monkeypatch.setattr(env.cfg.sim.nan_guard, "output_dir", str(tmp_path))
  counter = env.sim.data.ncollision
  saved = counter.clone()
  try:
    counter[:] = env.sim.wp_data.naconmax + 1
    with pytest.raises(RuntimeError, match="capacity overflow"):
      check_collision_capacity(env)
    report = json.loads((tmp_path / "capacity_failure.json").read_text())
    assert report["broadphase_pairs"]["counts"] == [env.sim.wp_data.naconmax + 1]
    assert (tmp_path / "capacity_failure.pt").is_file()
  finally:
    counter.copy_(saved)


@pytest.mark.slow
@torch.inference_mode()
def test_capture_rejects_raw_reward_before_sanitized_result(env, tmp_path, monkeypatch):
  import json

  from wbc_mjlab.diagnostics import install_failure_capture

  # Register restoration for the fixture's methods before installing wrappers.
  for owner, name in (
    (env.sim, "nan_guard"),
    (env.sim, "forward"),
    (env.sim, "sense"),
    (env.action_manager, "process_action"),
    (env.reward_manager, "compute"),
  ):
    monkeypatch.setattr(owner, name, getattr(owner, name))
  for name in env.reward_manager.active_terms:
    cfg = env.reward_manager.get_term_cfg(name)
    monkeypatch.setattr(cfg, "func", cfg.func)
  monkeypatch.setattr(
    env.cfg.sim,
    "nan_guard",
    NanGuardCfg(
      enabled=True,
      output_dir=str(tmp_path),
      buffer_size=2,
    ),
  )
  cfg = env.reward_manager.get_term_cfg("motion_global_root_pos")
  monkeypatch.setattr(
    cfg,
    "func",
    lambda env, **kwargs: torch.full(
      (env.num_envs,),
      float("nan"),
      device=env.device,
    ),
  )
  guard = install_failure_capture(env)
  env.reset()
  guard.capture(env.sim.data)
  with pytest.raises(FloatingPointError, match="raw_rewards"):
    env.reward_manager.compute(env.step_dt)
  metadata = json.loads(next(tmp_path.glob("physical_failure_*.json")).read_text())
  assert metadata["stage"] == "raw_rewards"
  assert metadata["faults"][0]["array"] == "motion_global_root_pos"
