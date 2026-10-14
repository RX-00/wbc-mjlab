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
