"""Bounded physical failure evidence, written before fatal NaN notification.

This captures inputs for investigation, not an exact closed-loop replay. The
standard mjlab state dump and fatal wrapper behavior are retained.
"""

import json
from contextlib import contextmanager
from datetime import datetime, timezone

import torch
from mjlab.utils.nan_guard import NanGuard

STATE_FIELDS = (
  "time",
  "qpos",
  "qvel",
  "act",
  "qacc",
  "qacc_warmstart",
  "ctrl",
  "qfrc_applied",
  "xfrc_applied",
  "mocap_pos",
  "mocap_quat",
)
INPUT_FIELDS = ("qpos", "qvel", "ctrl", "qfrc_applied", "xfrc_applied")
OUTPUT_FIELDS = ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata")
MODEL_FIELDS = (
  "body_mass",
  "body_inertia",
  "body_ipos",
  "body_iquat",
  "dof_armature",
  "dof_damping",
  "dof_frictionloss",
  "geom_friction",
  "geom_solref",
  "geom_solimp",
  "geom_margin",
  "geom_gap",
  "jnt_range",
  "actuator_gainprm",
  "actuator_biasprm",
  "actuator_forcerange",
  "actuator_ctrlrange",
)


class PhysicalFailureCapture(NanGuard):
  """Reuse the NaN guard ring; add actual physical inputs for every world."""

  def __init__(self, env):
    super().__init__(env.cfg.sim.nan_guard, env.num_envs, env.sim.mj_model)
    self.env = env
    self.buffer_bytes = 0
    self._reported_capacity = False

  def capture(self, data):
    super().capture(data)
    if not self.enabled:
      return
    state = self.buffer[-1]
    for name in STATE_FIELDS:
      if name not in state:
        state[name] = getattr(data, name).clone()
    self.buffer_bytes = (
      sum(
        value.numel() * value.element_size()
        for value in state.values()
        if isinstance(value, torch.Tensor)
      )
      * self.buffer_size
    )
    if not self._reported_capacity:
      self.output_dir.mkdir(parents=True, exist_ok=True)
      capacity = {
        "num_envs": self.num_envs,
        "steps": self.buffer_size,
        "seconds": self.buffer_size * self.env.physics_dt,
        "tensor_capacity_bytes": self.buffer_bytes,
        "fields": list(state),
        "scope": "All environments; current randomized model saved only on failure",
      }
      (self.output_dir / "capture_capacity.json").write_text(
        json.dumps(capacity, indent=2) + "\n"
      )
      self._reported_capacity = True

  def check(self, stage, tensors):
    if not self.enabled or self._dumped:
      return
    masks = {name: ~torch.isfinite(value) for name, value in tensors.items()}
    # One host synchronization on the ordinary path.
    if not torch.stack([mask.any() for mask in masks.values()]).any():
      return
    faults = []
    env_ids = set()
    for name, mask in masks.items():
      indices = mask.nonzero(as_tuple=False)
      if indices.numel():
        env_ids.update(indices[:, 0].unique().cpu().tolist())
        faults.append(
          {
            "array": name,
            "shape": list(tensors[name].shape),
            "count": len(indices),
            "indices": indices[:64].cpu().tolist(),
            "values": tensors[name][mask][:64].cpu().tolist(),
          }
        )
    ids = sorted(env_ids)
    selected = ids[: self.max_envs_to_dump]
    self.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    stem = self.output_dir / f"physical_failure_{stamp}"
    history = [
      {
        key: value[selected].cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
      }
      for state in self.buffer
    ]
    models = {}
    layouts = {}
    for name in MODEL_FIELDS:
      value = getattr(self.env.sim.model, name)
      cpu_shape = tuple(getattr(self.mj_model, name).shape)
      batched = tuple(value.shape[1:]) == cpu_shape
      if batched:
        models[name] = value[selected if value.shape[0] > 1 else [0]].cpu()
      else:
        models[name] = value.cpu()
      layouts[name] = {"shape": list(value.shape), "world_axis": 0 if batched else None}
    robot = self.env.scene["robot"]
    context = {
      "raw_action": self.env.action_manager.action[selected].cpu(),
      "joint_position_target": robot.data.joint_pos_target[selected].cpu(),
      "joint_velocity_target": robot.data.joint_vel_target[selected].cpu(),
      "episode_length": self.env.episode_length_buf[selected].cpu(),
    }
    motion = self.env.command_manager.get_term("motion")
    for name in ("time_steps", "assist_force_w", "assist_torque_w"):
      context[name] = getattr(motion, name)[selected].cpu()
    actuators = []
    for actuator in robot.actuators:
      actuators.append(
        {
          "target_names": actuator._target_names,
          **{
            name: getattr(actuator, name)[selected].cpu()
            for name in (
              "stiffness",
              "damping",
              "force_limit",
              "motor_effort",
              "_friction_static",
              "_friction_dynamic",
              "_activation_vel",
            )
          },
        }
      )
    payload = {
      "history": history,
      "detected": {name: value[selected].cpu() for name, value in tensors.items()},
      "model": models,
      "context": context,
      "actuators": actuators,
    }
    torch.save(payload, stem.with_suffix(".pt"))
    metadata = {
      "stage": stage,
      "step": self.step_counter,
      "faults": faults,
      "environment_ids": ids,
      "saved_environment_ids": selected,
      "history_steps": len(history),
      "ring_capacity_bytes": self.buffer_bytes,
      "physics_dt": self.env.physics_dt,
      "model_layouts": layouts,
      "limitations": [
        "Current randomized model values; not a history of model mutations",
        "No optimizer/RNG/history-normalizer state or solver matrices",
        "Not a complete closed-loop replay bundle",
      ],
    }
    # Nonfinite values are encoded as strings to keep metadata valid JSON.
    for fault in faults:
      fault["values"] = [str(value) for value in fault["values"]]
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Only now may the standard guard print the message that kills the child.
    self._dump_buffer(ids)
    self._dumped = True
    raise FloatingPointError(f"Nonfinite {stage}; physical evidence: {stem}")

  @contextmanager
  def watch(self, data):
    self.capture(data)
    self.check("pre_step", {name: getattr(data, name) for name in INPUT_FIELDS})
    yield
    self.check("post_step", {name: getattr(data, name) for name in OUTPUT_FIELDS})


def install_failure_capture(env):
  """Install after environment initialization, before collecting rollouts."""
  if not env.cfg.sim.nan_guard.enabled:
    return None
  if isinstance(env.sim.nan_guard, PhysicalFailureCapture):
    return env.sim.nan_guard
  guard = PhysicalFailureCapture(env)
  env.sim.nan_guard = guard
  for name in ("forward", "sense"):
    original = getattr(env.sim, name)

    def checked(*args, _original=original, _stage=name, **kwargs):
      result = _original(*args, **kwargs)
      guard.check(
        _stage, {field: getattr(env.sim.data, field) for field in OUTPUT_FIELDS}
      )
      return result

    setattr(env.sim, name, checked)

  process_action = env.action_manager.process_action

  def checked_action(action):
    guard.check("policy_action", {"action": action})
    process_action(action)
    guard.check(
      "processed_actions",
      {
        # mjlab 1.6 BaseAction exposes this buffer only as a private field;
        # both WBC position-action implementations use it in apply_actions.
        name: env.action_manager.get_term(name)._processed_actions
        for name in env.action_manager.active_terms
      },
    )

  env.action_manager.process_action = checked_action

  # mjlab sanitizes reward values inside compute. Retain the raw tensors and
  # reject them before compute returns to the learner (one batched check).
  raw_rewards = {}

  class RecordedReward:
    def __init__(self, name, function):
      self.name, self.function = name, function

    def __call__(self, *args, **kwargs):
      value = self.function(*args, **kwargs)
      raw_rewards[self.name] = value
      return value

    def __getattr__(self, name):
      return getattr(self.function, name)

  for name in env.reward_manager.active_terms:
    cfg = env.reward_manager.get_term_cfg(name)
    cfg.func = RecordedReward(name, cfg.func)
  compute_rewards = env.reward_manager.compute

  def checked_rewards(*args, **kwargs):
    raw_rewards.clear()
    result = compute_rewards(*args, **kwargs)
    if raw_rewards:
      guard.check("raw_rewards", raw_rewards)
    return result

  env.reward_manager.compute = checked_rewards
  return guard
