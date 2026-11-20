"""Deterministic, unassisted training-set tracking evaluation (no viewer).

Example: python -m wbc_mjlab.scripts.evaluate --task Wbc-G1-SE
  --checkpoint /runs/train/.../model_999.pt --motion-file data/g1/samples
  --output /runs/evaluation/pilot-g1-se
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.actuator.pd_actuator import IdealPdActuator
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude
from onnx.reference import ReferenceEvaluator
from onnx.reference.ops._op import OpRunUnaryNum

from wbc_mjlab.env.mdp.torque_envelope import tapered_peak
from wbc_mjlab.robots.g1.actuators import UnitreeActuator, UnitreeActuatorCfg
from wbc_mjlab.robots.g1.constants import G1_FOOT_SITE_NAMES
from wbc_mjlab.tasks import prepare_wbc_run

PROTOCOL = {
  "version": 3,
  "seed": 42,
  "control_period_s": 0.02,
  "collision_capacity": {"nconmax": 128, "njmax": 1024, "contact_sensor_maxmatch": 128},
  "segment_fractions": [0.25, 0.5, 0.75],
  "segment_steps": 250,
  "failure": {"anchor_height_error_m": 0.25, "gravity_z_error": 0.8},
  "onnx_atol": 1e-5,
  "onnx_rtol": 1e-4,
  "initialization": "Reference pose (joint positions clipped to soft limits), reference velocities; reference position targets, zero velocity/feedforward targets and warmstart before forward",
  "randomization": "None; model nominal friction/inertia/COM, zero encoder bias, no pushes",
  "assistance": "Command disabled, all events removed, gain and applied wrench verified zero",
  "sensing": "Ideal simulator state, actor corruption disabled",
  "coverage": "Training-set tracking; errors include samples up to and including failure",
  "metrics": "Post-control state against newly advanced reference; mean over joints/bodies then time",
  "tracking_targets": {
    "completion_rate": 0.9,
    "body_global_m": 0.15,
    "joint_position_rmse_rad": 0.3,
  },
  "target_scope": "Initial nominal tracking targets, reported separately from numerical stability",
}


class Elu(OpRunUnaryNum):
  """Reference ELU without evaluating exp on the unused positive branch.

  ONNX's NumPy implementation uses where, which eagerly computes exp(x) even
  for large positive inputs whose specified output is simply x. Keep overflow
  reporting fatal by avoiding that unnecessary operation, not its warning.
  """

  op_domain = ""

  def _run(self, x, alpha=None):
    alpha = self.alpha if alpha is None else alpha
    result = x.copy()
    negative = x <= 0
    result[negative] = alpha * (np.exp(x[negative]) - 1)
    return (result,)


def load_onnx_policy(path):
  return ReferenceEvaluator(str(path), new_ops=[Elu])


class HistoricalActuator(UnitreeActuator):
  """Evaluation-only reproduction of the old bypassed-envelope compute path."""

  def compute(self, cmd):
    effort = IdealPdActuator.compute(self, cmd)
    self.motor_effort.copy_(effort)
    return effort - (
      self._friction_static * torch.tanh(cmd.vel / self._activation_vel)
      + self._friction_dynamic * cmd.vel
    )


class HistoricalActuatorCfg(UnitreeActuatorCfg):
  def build(self, entity, target_ids, target_names):
    return HistoricalActuator(self, entity, target_ids, target_names)


def sha256(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def check_collision_capacity(env):
  """Reject truncated collision/constraint work even without a log wrapper."""
  data, allocation = env.sim.data, env.sim.wp_data
  counters = {
    "broadphase_pairs": (data.ncollision, allocation.naconmax),
    "contacts": (data.nacon, allocation.naconmax),
    "constraints": (data.nefc, allocation.njmax),
  }
  if not torch.stack(
    [(value > limit).any() for value, limit in counters.values()]
  ).any():
    return
  output = Path(env.cfg.sim.nan_guard.output_dir)
  output.mkdir(parents=True, exist_ok=True)
  report = {
    key: {"counts": value.cpu().tolist(), "capacity": limit}
    for key, (value, limit) in counters.items()
  }
  (output / "capacity_failure.json").write_text(json.dumps(report, indent=2) + "\n")
  torch.save(
    {name: getattr(data, name).cpu() for name in ("qpos", "qvel", "ctrl", "qacc")},
    output / "capacity_failure.pt",
  )
  raise RuntimeError(f"Simulator capacity overflow; evidence: {output}")


def trial_schedule(motion):
  trials = []
  for clip, (name, start, length) in enumerate(
    zip(
      motion.segment_names,
      motion.segment_start_idx.tolist(),
      motion.segment_length.tolist(),
      strict=True,
    )
  ):
    if length < 2:
      raise ValueError(f"Clip {name} has no transition to evaluate")
    trials.append(
      dict(clip=clip, name=name, suite="full", start=start, steps=length - 1)
    )
    for fraction in PROTOCOL["segment_fractions"]:
      offset = min(int((length - 1) * fraction), length - 2)
      trials.append(
        dict(
          clip=clip,
          name=name,
          suite=f"segment_{fraction}",
          start=start + offset,
          steps=min(PROTOCOL["segment_steps"], length - offset - 1),
        )
      )
  return trials


def configure_evaluation(cfg, motion_file, output, physics_dt):
  cfg.scene.num_envs = 1
  cfg.seed = PROTOCOL["seed"]
  cfg.auto_reset = False
  cfg.terminations = {}  # Common rule below, evaluated on refreshed terminal state.
  cfg.events = {}
  cfg.curriculum = {}
  cfg.sim.mujoco.timestep = physics_dt
  for sensor in cfg.scene.sensors:
    if getattr(sensor, "history_length", 0):
      sensor.history_length = round(sensor.history_length * 0.005 / physics_dt)
  cfg.decimation = round(PROTOCOL["control_period_s"] / physics_dt)
  if abs(cfg.decimation * physics_dt - 0.02) > 1e-12:
    raise ValueError("Physics timestep must divide the 50 Hz control period")
  cfg.sim.njmax = 1024
  # A single world's contact pool cannot borrow capacity from other worlds.
  # The heuristic pool overflowed during full-clip pilot evaluations.
  cfg.sim.nconmax = PROTOCOL["collision_capacity"]["nconmax"]
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nan_guard.enabled = True
  cfg.sim.nan_guard.output_dir = str(output / "diagnostics")
  cfg.sim.nan_guard.buffer_size = round(0.16 / physics_dt)
  for group in cfg.observations.values():
    group.enable_corruption = False
    group.nan_policy = "error"
  motion = cfg.commands["motion"]
  motion.motion_file = str(motion_file)
  motion.pose_range = {}
  motion.velocity_range = {}
  motion.joint_position_range = (0.0, 0.0)
  motion.assistive_wrench_enabled = False
  motion.rsi.sampling_mode = "start"
  # Trial starts are explicit; adaptive-bin occupancy must not reject short clips.
  motion.rsi.min_bin_span_ratio = 0.0
  motion.rsi.persist_failure_levels = False


def tracking_metrics(motion):
  reference = motion.body_pos_w - motion.anchor_pos_w[:, None]
  actual = motion.robot_body_pos_w - motion.robot_anchor_pos_w[:, None]
  nbody = reference.shape[1]
  reference_local = quat_apply_inverse(
    motion.anchor_quat_w[:, None].expand(-1, nbody, -1), reference
  )
  actual_local = quat_apply_inverse(
    motion.robot_anchor_quat_w[:, None].expand(-1, nbody, -1), actual
  )
  values = {
    "anchor_drift_m": (motion.anchor_pos_w - motion.robot_anchor_pos_w)
    .norm(dim=-1)
    .mean(),
    "body_global_m": (motion.body_pos_w - motion.robot_body_pos_w).norm(dim=-1).mean(),
    "body_root_relative_m": (reference_local - actual_local).norm(dim=-1).mean(),
    "body_orientation_rad": quat_error_magnitude(
      motion.body_quat_w, motion.robot_body_quat_w
    ).mean(),
    "joint_position_rmse_rad": (motion.joint_pos - motion.robot_joint_pos)
    .square()
    .mean()
    .sqrt(),
    "joint_velocity_rmse_rad_s": (motion.joint_vel - motion.robot_joint_vel)
    .square()
    .mean()
    .sqrt(),
    "body_linear_velocity_m_s": (motion.body_lin_vel_w - motion.robot_body_lin_vel_w)
    .norm(dim=-1)
    .mean(),
    "body_angular_velocity_rad_s": (motion.body_ang_vel_w - motion.robot_body_ang_vel_w)
    .norm(dim=-1)
    .mean(),
    "anchor_linear_velocity_m_s": (
      motion.anchor_lin_vel_w - motion.robot_anchor_lin_vel_w
    )
    .norm(dim=-1)
    .mean(),
  }
  return {name: value.item() for name, value in values.items()}


class PhysicalMetrics:
  """Single-world physical telemetry; active torque and drag remain separate.

  Work uses pre-integration torque/velocity at every physics substep. Foot
  slip, net contact force and hard-limit violations are sampled at 50 Hz.
  Penetration/contact-count maxima use every physics substep's contact solve.
  """

  def __init__(self, env):
    assert env.num_envs == 1
    self.env = env
    self.robot = env.scene["robot"]
    self.site_ids = self.robot.find_sites(G1_FOOT_SITE_NAMES, preserve_order=True)[0]
    self.original_step = env.sim.step
    env.sim.step = self.step
    self.reset()

  def reset(self):
    names = (
      "motor_positive_work_j",
      "motor_braking_work_j",
      "friction_dissipation_j",
      "motor_torque_peak_nm",
      "motor_power_peak_w",
      "joint_speed_peak_rad_s",
      "saturated_joint_samples",
      "overspeed_joint_samples",
      "motor_joint_samples",
      "penetration_peak_m",
      "contacts_peak",
      "foot_slip_speed_sum_m_s",
      "foot_contact_samples",
      "foot_net_force_peak_n",
      "joint_limit_samples",
      "joint_samples",
    )
    self.values = {name: torch.zeros((), device=self.env.device) for name in names}

  def step(self):
    v = self.values
    for actuator in self.robot.actuators:
      velocity = self.robot.data.joint_vel[:, actuator.target_ids]
      torque = actuator.motor_effort
      envelope = actuator._envelope
      peak = torch.where(velocity * torque > 0, envelope.y1, envelope.y2)
      limit = torch.minimum(
        tapered_peak(velocity.abs(), peak, envelope.x1, envelope.x2),
        actuator.force_limit,
      )
      v["saturated_joint_samples"] += (
        (limit > 1e-6) & (torque.abs() >= 0.99 * limit)
      ).sum()
      v["overspeed_joint_samples"] += (velocity.abs() >= envelope.x2).sum()
      v["motor_joint_samples"] += torque.numel()
      power = torque * velocity
      passive = self.env.sim.data.ctrl[:, actuator.global_ctrl_ids] - torque
      v["motor_positive_work_j"] += power.clamp_min(0).sum() * self.env.physics_dt
      v["motor_braking_work_j"] += (-power).clamp_min(0).sum() * self.env.physics_dt
      v["friction_dissipation_j"] += (-passive * velocity).sum() * self.env.physics_dt
      for key, value in (
        ("motor_torque_peak_nm", torque.abs().max()),
        ("motor_power_peak_w", power.abs().max()),
        ("joint_speed_peak_rad_s", velocity.abs().max()),
      ):
        v[key] = torch.maximum(v[key], value)
    self.original_step()
    check_collision_capacity(self.env)
    data = self.env.sim.data
    distances = data.contact.dist[:]
    valid = torch.arange(distances.numel(), device=self.env.device) < data.nacon[0]
    penetration = torch.where(valid, -distances, 0).max().clamp_min(0)
    v["penetration_peak_m"] = torch.maximum(v["penetration_peak_m"], penetration)
    v["contacts_peak"] = torch.maximum(v["contacts_peak"], data.nacon[0])

  def control_sample(self):
    v = self.values
    contact = self.env.scene["feet_ground_contact"].data
    found = contact.found > 0
    speed = self.robot.data.site_lin_vel_w[:, self.site_ids, :2].norm(dim=-1)
    assert found.shape == speed.shape
    v["foot_slip_speed_sum_m_s"] += (speed * found).sum()
    v["foot_contact_samples"] += found.sum()
    v["foot_net_force_peak_n"] = torch.maximum(
      v["foot_net_force_peak_n"], contact.force.norm(dim=-1).max()
    )
    limits = self.robot.data.joint_pos_limits
    position = self.robot.data.joint_pos
    v["joint_limit_samples"] += (
      (position < limits[..., 0]) | (position > limits[..., 1])
    ).sum()
    v["joint_samples"] += position.numel()

  def report(self):
    result = {name: value.item() for name, value in self.values.items()}
    for output, numerator, denominator in (
      ("motor_saturation_fraction", "saturated_joint_samples", "motor_joint_samples"),
      ("overspeed_fraction", "overspeed_joint_samples", "motor_joint_samples"),
      ("foot_slip_m_s", "foot_slip_speed_sum_m_s", "foot_contact_samples"),
      ("joint_limit_violation_fraction", "joint_limit_samples", "joint_samples"),
    ):
      result[output] = (
        result[numerator] / result[denominator] if result[denominator] else None
      )
    return {
      key: value if value is None or np.isfinite(value) else str(value)
      for key, value in result.items()
    }


def fallen(motion):
  height = (motion.anchor_pos_w[:, 2] - motion.robot_anchor_pos_w[:, 2]).abs()
  gravity = motion.robot.data.gravity_vec_w
  reference = quat_apply_inverse(motion.anchor_quat_w, gravity)
  actual = quat_apply_inverse(motion.robot_anchor_quat_w, gravity)
  return bool(
    (height > PROTOCOL["failure"]["anchor_height_error_m"]).any()
    or (
      (reference[:, 2] - actual[:, 2]).abs() > PROTOCOL["failure"]["gravity_z_error"]
    ).any()
  )


def initialize_trial(env, trial):
  env.reset()
  motion = env.command_manager.get_term("motion")
  ids = torch.tensor([0], device=env.device)
  motion.reset_to_frame(ids, trial["start"])
  assert int(motion.trajectory_ids[0]) == trial["clip"]
  motion._seed_default_relative_action(ids)
  # Explicit physical inputs before initial sensing, independent of the
  # action representation and the first clip used internally by env.reset().
  motion.robot.set_joint_position_target(motion.joint_pos)
  motion.robot.set_joint_velocity_target(torch.zeros_like(motion.joint_vel))
  motion.robot.set_joint_effort_target(torch.zeros_like(motion.joint_vel))
  env.scene.write_data_to_sim()
  env.sim.data.qacc_warmstart[:] = 0.0
  env.sim.forward()
  motion.update_relative_body_poses()
  env.sim.sense()
  check_collision_capacity(env)
  env.observation_manager.reset(ids)
  env.observation_manager.compute(update_history=True)
  assert not motion.episode_assist_gain.any()
  assert not env.sim.data.xfrc_applied.any()


def summarize(trials):
  summaries = {}
  for suite in sorted({trial["suite"] for trial in trials}):
    rows = [trial for trial in trials if trial["suite"] == suite]
    summaries[suite] = {
      "planned": len(rows),
      "completed": sum(t["status"] == "completed" for t in rows),
      "falls": sum(t["status"] == "fall" for t in rows),
      "numerical_failures": sum(t["status"] == "numerical_failure" for t in rows),
      "completion_rate": sum(t["status"] == "completed" for t in rows) / len(rows),
      "observed_steps": sum(t.get("observed_steps", 0) for t in rows),
      "planned_steps": sum(t["steps"] for t in rows),
    }
    measured = [row for row in rows if row.get("metrics")]
    summaries[suite]["trials_with_metrics"] = len(measured)
    summaries[suite]["mean_trial_metrics"] = (
      {
        key: float(np.mean([row["metrics"][key] for row in measured]))
        for key in measured[0]["metrics"]
      }
      if measured
      else {}
    )
  return summaries


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--motion-file", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--physics-dt", type=float, choices=[0.005, 0.0025, 0.001], default=0.005
  )
  parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
  parser.add_argument(
    "--actuator-model",
    choices=["corrected", "historical_unbounded"],
    default="corrected",
  )
  parser.add_argument(
    "--onnx-file",
    type=Path,
    help="Also validate this existing export against the selected checkpoint",
  )
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  started = time.monotonic()
  args.output.mkdir(parents=True, exist_ok=False)
  prepare_wbc_run(task_id=args.task)
  cfg = load_env_cfg(args.task, play=True)
  configure_evaluation(cfg, args.motion_file, args.output, args.physics_dt)
  if args.actuator_model == "historical_unbounded":
    robot = cfg.scene.entities["robot"]
    robot.articulation = replace(
      robot.articulation,
      actuators=tuple(
        HistoricalActuatorCfg(**asdict(actuator))
        for actuator in robot.articulation.actuators
      ),
    )
  env = ManagerBasedRlEnv(cfg, device=args.device)
  trials = []
  result = {
    "task": args.task,
    "checkpoint": str(args.checkpoint),
    "checkpoint_sha256": sha256(args.checkpoint),
    "protocol": PROTOCOL,
    "physics_dt": args.physics_dt,
    "backend": args.backend,
    "versions": {
      name: importlib.metadata.version(name)
      for name in ("mjlab", "mujoco", "mujoco-warp", "torch", "onnx")
    },
  }
  try:
    agent = load_rl_cfg(args.task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(args.task)(wrapped, asdict(agent), device=args.device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    runner.export_policy_to_onnx(str(args.output), "policy.onnx")
    result["export_metadata"] = runner.build_policy_export_metadata(
      str(args.checkpoint)
    )
    exported = load_onnx_policy(args.output / "policy.onnx")
    existing_export = load_onnx_policy(args.onnx_file) if args.onnx_file else None
    if args.onnx_file:
      result["input_export"] = {
        "path": str(args.onnx_file),
        "sha256": sha256(args.onnx_file),
      }
    assert exported.input_names == ["obs"]
    motion = env.command_manager.get_term("motion")
    physical = PhysicalMetrics(env)
    result["body_names"] = list(motion.cfg.body_names)
    result["joint_names"] = list(motion.robot.joint_names)
    result["nominal_geom_friction"] = env.sim.mj_model.geom_friction.tolist()
    # Preserve the exact nominal model for rendering saved states without
    # re-simulating the trajectory or loading a future source revision.
    with tempfile.TemporaryDirectory() as temporary:
      model_path = Path(temporary) / "model.mjb"
      mujoco.mj_saveModel(env.sim.mj_model, str(model_path), None)
      with (
        model_path.open("rb") as source,
        gzip.open(args.output / "model.mjb.gz", "wb") as destination,
      ):
        shutil.copyfileobj(source, destination)
    result["model_sha256"] = sha256(args.output / "model.mjb.gz")
    paths = (
      sorted(args.motion_file.glob("npz/*.npz"))
      if args.motion_file.is_dir()
      else [args.motion_file]
    )
    result["data_sha256"] = {str(path): sha256(path) for path in paths}
    trials = [dict(t, status="not_started") for t in trial_schedule(motion.motion)]
    result["trials"] = trials
    (args.output / "protocol.json").write_text(json.dumps(result, indent=2) + "\n")
    initial_states = []
    initial_controls = []
    parity_obs, parity_actions = [], []
    with torch.inference_mode():
      for trial_id, trial in enumerate(trials):
        initialize_trial(env, trial)
        initial_states.append(
          torch.cat([env.sim.data.qpos[0], env.sim.data.qvel[0]]).cpu().numpy()
        )
        initial_controls.append(env.sim.data.ctrl[0].cpu().numpy().copy())
        samples = []
        trace = {name: [] for name in ("qpos", "qvel", "action", "frame")}
        physical.reset()
        trial.update(status="running", observed_steps=0)
        loop_started = time.monotonic()
        try:
          for step in range(trial["steps"]):
            obs = wrapped.get_observations()
            action = policy(obs)
            if not torch.isfinite(action).all():
              raise FloatingPointError("Nonfinite deterministic policy action")
            # Ordinary and start/end boundary observations, plus every ONNX control.
            if step % 50 == 0 or step == trial["steps"] - 1 or args.backend == "onnx":
              exported_action = exported.run(None, {"obs": obs["actor"].cpu().numpy()})[
                0
              ]
              np.testing.assert_allclose(
                exported_action,
                action.cpu().numpy(),
                atol=PROTOCOL["onnx_atol"],
                rtol=PROTOCOL["onnx_rtol"],
              )
              if existing_export is not None:
                existing_actions = existing_export.run(
                  None, {"obs": obs["actor"].cpu().numpy()}
                )[0]
                np.testing.assert_allclose(
                  existing_actions,
                  action.cpu().numpy(),
                  atol=PROTOCOL["onnx_atol"],
                  rtol=PROTOCOL["onnx_rtol"],
                )
              parity_obs.append(obs["actor"].cpu().numpy())
              parity_actions.append(action.cpu().numpy())
              if args.backend == "onnx":
                action = torch.from_numpy(exported_action).to(env.device)
            wrapped.step(action)
            assert int(motion.time_steps[0]) == trial["start"] + step + 1, (
              "Unexpected wrap/reset"
            )
            assert not motion.episode_assist_gain.any()
            assert not motion.assist_force_w.any() and not motion.assist_torque_w.any()
            assert not env.sim.data.xfrc_applied.any(), (
              "Evaluation assistance/external wrench is nonzero"
            )
            sample = tracking_metrics(motion)
            physical.control_sample()
            if not all(np.isfinite(value) for value in sample.values()):
              raise FloatingPointError("Nonfinite tracking metric")
            samples.append(sample)
            trace["qpos"].append(env.sim.data.qpos[0].cpu().numpy().copy())
            trace["qvel"].append(env.sim.data.qvel[0].cpu().numpy().copy())
            trace["action"].append(action[0].cpu().numpy().copy())
            trace["frame"].append(int(motion.time_steps[0]))
            trial["observed_steps"] += 1
            if fallen(motion):
              trial["status"] = "fall"
              break
          else:
            trial["status"] = "completed"
        except FloatingPointError as error:
          trial.update(status="numerical_failure", error=str(error))
          raise
        except Exception as error:
          trial.update(status="evaluation_error", error=str(error))
          raise
        finally:
          trial["loop_wall_seconds"] = time.monotonic() - loop_started
          trial["observed_seconds"] = trial["observed_steps"] * env.step_dt
          trial["metrics"] = (
            {key: float(np.mean([s[key] for s in samples])) for key in samples[0]}
            if samples
            else {}
          )
          trial["terminal_qpos"] = [
            value if np.isfinite(value) else str(value)
            for value in env.sim.data.qpos[0].cpu().tolist()
          ]
          trial["physical"] = physical.report()
          trial["trace"] = f"trial_{trial_id:03d}_trace.npz"
          np.savez_compressed(
            args.output / trial["trace"],
            **{name: np.asarray(values) for name, values in trace.items()},
          )
          np.savez_compressed(
            args.output / "initial_states.npz",
            state=np.stack(initial_states),
            ctrl=np.stack(initial_controls),
          )
          with (args.output / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(trial, allow_nan=False) + "\n")
    np.savez_compressed(
      args.output / "initial_states.npz",
      state=np.stack(initial_states),
      ctrl=np.stack(initial_controls),
    )
    np.savez_compressed(
      args.output / "export_parity.npz",
      obs=np.concatenate(parity_obs),
      actions=np.concatenate(parity_actions),
    )
    result["export_parity"] = {
      "samples": len(parity_obs),
      "passed": True,
      "engine": "onnx.reference.ReferenceEvaluator",
      "elu_implementation": "negative_branch_only_v1",
      "existing_export_checked": existing_export is not None,
    }
  finally:
    result["elapsed_seconds"] = time.monotonic() - started
    result["actuator_model"] = args.actuator_model
    result["summary"] = summarize(trials)
    (args.output / "results.json").write_text(
      json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    env.close()


if __name__ == "__main__":
  main()
