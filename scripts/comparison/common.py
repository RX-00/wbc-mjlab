"""Closed-loop cross-pipeline trials on the archived v12 nominal G1 in CPU MuJoCo."""

import argparse
import json
import re
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch
import yaml
from convert_reference import load_model
from planner import Planner
from scipy.spatial.transform import Rotation


def rot(q):
  return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def matched(names, groups, key):
  values = []
  for name in names:
    matches = [
      float(g[key]) for g in groups if any(re.fullmatch(p, name) for p in g["patterns"])
    ]
    if len(matches) != 1:
      raise ValueError(f"Expected one {key} for {name}, got {matches}")
    values.append(matches[0])
  return np.array(values)


class Actor:
  def __init__(self, folder):
    self.history = None
    if (folder / "contract.json").exists():
      contract = json.loads((folder / "contract.json").read_text())
      self.task = contract["task"]
      self.names = contract["state_joint_names"]
      if contract["names"] != self.names:
        raise ValueError("SuperMimic action and sensor orders differ")
      self.scale, self.default = map(np.array, (contract["scale"], contract["offset"]))
      self.sm = torch.jit.load(str(folder / "actor.ts"), map_location="cpu").eval()
      self.preview = True
      self.widths = contract["widths"]
      self.length = contract["history_length"]
      self.gains = contract["gains"]
      self.offsets = np.arange(10) * 5
      self.identity = (
        contract["sha256"]["actor.pt"]
        if "actor.pt" in contract["sha256"]
        else contract["sha256"]
      )
    else:
      results = json.loads((folder / "evaluation/results.json").read_text())
      self.task = results["task"]
      c = results["export_metadata"]
      self.sm = None
      self.names = c["joint_names"]
      self.scale, self.default = map(
        np.array, (c["action_scale"], c["default_joint_pos"])
      )
      self.terms = c["observation_names"]
      if (
        self.task not in ("Wbc-G1", "Wbc-G1-Zest", "Wbc-G1-Preview")
        or c["action_mode"] != "reference_residual"
      ):
        raise ValueError(f"{self.task} requires additional deployed state estimation")
      options = ort.SessionOptions()
      options.intra_op_num_threads = 1
      options.inter_op_num_threads = 1
      export = folder / "evaluation/policy.onnx"
      if not export.is_file():
        export = folder / "params/policy.onnx"
      self.model = ort.InferenceSession(
        str(export), options, providers=["CPUExecutionProvider"]
      )
      self.identity = results["checkpoint_sha256"]
      self.offsets = np.array([0])
      self.gains = None
      self.preview = self.task == "Wbc-G1-Preview"
      if self.preview:
        fields = [
          "projected_gravity",
          "base_ang_vel",
          "joint_pos",
          "joint_vel",
          "actions",
          "ref_motion_window",
        ]
        self.length = c["sensor_history_length"]
        self.widths = [3, 3, 29, 29, 29]
        self.offsets = np.array(c["preview_offsets"])
        if (
          self.terms != fields
          or self.length != 10
          or c["wbc_command_dim"] != 640
          or self.offsets.tolist() != list(range(0, 50, 5))
          or self.model.get_inputs()[0].shape[-1]
          != self.length * sum(self.widths) + 640
        ):
          raise ValueError("Unexpected WBC preview layout")

  def start(self):
    self.history = None
    self.last = np.zeros(29)

  def inputs(self, targets, torso, attitude, gyro, position, velocity):
    gravity = np.array([0.0, 0.0, -1.0])
    terms = dict(
      projected_gravity=attitude.inv().apply(gravity),
      base_ang_vel=gyro,
      joint_pos=position - self.default,
      joint_vel=velocity,
      actions=self.last,
    )
    if self.preview:
      now = np.concatenate(list(terms.values()))
      self.history = (
        np.tile(now, (self.length, 1))
        if self.history is None
        else np.concatenate((self.history[1:], now[None]))
      )
      # Native export takes term-major history followed by interleaved q,dq and 6D orientations.
      target_rot = rot(targets["body_quat_w"][:, 0])
      rel = (attitude.inv() * target_rot).as_matrix()[..., :2]
      content = np.concatenate((targets["joint_pos"], targets["joint_vel"]), axis=-1)
      parts = np.split(self.history, np.cumsum(self.widths)[:-1], axis=1)
      return np.concatenate(
        [*(part.ravel() for part in parts), content.ravel(), rel.ravel()]
      ).astype(np.float32)[None]
    ref_rot = rot(targets["body_quat_w"][0, torso])
    terms.update(
      ref_base_height=targets["body_pos_w"][0, torso, 2:],
      ref_gravity_b=ref_rot.inv().apply(gravity),
      ref_base_lin_vel_b=ref_rot.inv().apply(targets["body_lin_vel_w"][0, torso]),
      ref_base_ang_vel_b=ref_rot.inv().apply(targets["body_ang_vel_w"][0, torso]),
      ref_joint_pos=targets["joint_pos"][0],
    )
    return np.concatenate([terms[t] for t in self.terms]).astype(np.float32)[None]

  def predict(self, obs):
    if self.sm is not None:
      with torch.inference_mode():
        return self.sm(torch.from_numpy(obs))[0].numpy()
    return self.model.run(None, {"obs": obs})[0][0]


class Arena:
  def __init__(self, baseline):
    self.m = load_model(baseline / "evaluation/model.mjb.gz")
    if self.m.opt.timestep != 0.0025:
      raise ValueError("Expected corrected 2.5 ms profile")
    self.joints = [self.m.joint(i).name.removeprefix("robot/") for i in range(1, 30)]
    self.ids = np.array(
      [i for i in range(self.m.nbody) if self.m.body(i).name.startswith("robot/")]
    )
    cfg = yaml.load((baseline / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    motors = cfg["scene"]["entities"]["robot"]["articulation"]["actuators"]
    self.groups = [dict(g, patterns=g["target_names_expr"]) for g in motors]
    self.ctrl = [self.m.actuator("robot/" + n).id for n in self.joints]
    self.torso = list(self.ids).index(self.m.body("robot/torso_link").id)
    for key in ("x1", "x2", "y1", "y2", "fs", "fd", "va"):
      setattr(self, key, matched(self.joints, self.groups, key))
    self.lower, self.upper = self.m.jnt_range[1:].T
    self.d = mujoco.MjData(self.m)

  def trial(self, actor, planner, gyro_bias):
    m, d = self.m, self.d
    if actor.names != self.joints:
      raise ValueError("Policy/reference/model joint order mismatch")
    if actor.gains:
      kp, kd = [matched(self.joints, actor.gains, k) for k in ("kp", "kd")]
    else:
      kp, kd = [matched(self.joints, self.groups, k) for k in ("stiffness", "damping")]
    r = planner.original
    attitude = rot(r["body_quat_w"][0, 0])  # explicit commanded initial alignment
    mujoco.mj_resetData(m, d)
    d.qpos[:3], d.qpos[3:7] = r["body_pos_w"][0, 0], r["body_quat_w"][0, 0]
    mid = (self.upper + self.lower) / 2
    d.qpos[7:] = np.clip(
      r["joint_pos"][0], mid + 0.9 * (self.lower - mid), mid + 0.9 * (self.upper - mid)
    )
    d.qvel[:3] = r["body_lin_vel_w"][0, 0]
    d.qvel[3:6] = attitude.inv().apply(r["body_ang_vel_w"][0, 0])
    d.qvel[6:] = r["joint_vel"][0]
    mujoco.mj_forward(m, d)
    actor.start()
    trace = {
      k: []
      for k in (
        "qpos",
        "ref_qpos",
        "path_m",
        "heading_deg",
        "pose_m",
        "joint_rmse_rad",
        "action",
        "saturation",
        "inference_ms",
      )
    }
    initial = d.qpos.copy()
    initial_velocity = d.qvel.copy()
    gyro = d.sensor("robot/imu_ang_vel").data
    status = "completed"
    for step in range(planner.length - 1):
      planner.update(step)
      targets = planner.targets(step, actor.offsets)
      t0 = time.perf_counter_ns()
      obs = actor.inputs(
        targets, self.torso, attitude, gyro + gyro_bias, d.qpos[7:], d.qvel[6:]
      )
      action = actor.predict(obs)
      elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
      if not np.isfinite(action).all():
        raise FloatingPointError("Nonfinite policy output")
      actor.last = action.copy()
      target = (
        actor.default if actor.sm is not None else targets["joint_pos"][0]
      ) + actor.scale * action
      sat = 0
      for _ in range(8):
        dq = d.qvel[6:]
        effort = kp * (target - d.qpos[7:]) - kd * dq
        peak = np.where(dq * effort > 0, self.y1, self.y2)
        limit = peak * np.clip((self.x2 - np.abs(dq)) / (self.x2 - self.x1), 0, 1)
        sat += np.count_nonzero(np.abs(effort) > limit)
        d.ctrl[self.ctrl] = (
          np.clip(effort, -limit, limit)
          - self.fs * np.tanh(dq / self.va)
          - self.fd * dq
        )
        mujoco.mj_step(m, d)
        attitude *= Rotation.from_rotvec((gyro.copy() + gyro_bias) * m.opt.timestep)
      if not np.isfinite(d.qpos).all() or np.any(
        d.warning.number[
          [
            int(mujoco.mjtWarning.mjWARN_BADQPOS),
            int(mujoco.mjtWarning.mjWARN_BADQVEL),
            int(mujoco.mjtWarning.mjWARN_BADQACC),
          ]
        ]
      ):
        raise FloatingPointError("Numerical failure in common simulator")
      mujoco.mj_forward(m, d)
      idx = step + 1
      truth = planner.intent(idx)
      p = truth["body_pos_w"][idx]
      q = truth["body_quat_w"][idx]
      real = d.xpos[self.ids]
      root = rot(d.qpos[3:7])
      desired = rot(q[0])
      # Heading rotates about gravity. Pose uses pelvis frames so heading drift
      # is reported separately. Keep root height in the fall rule below.
      yaw = np.arctan2(*root.as_matrix()[:2, 0][::-1]) - np.arctan2(
        *desired.as_matrix()[:2, 0][::-1]
      )
      vals = dict(
        qpos=d.qpos.copy(),
        ref_qpos=np.r_[p[0], q[0], truth["joint_pos"][idx]],
        path_m=np.linalg.norm(real[0, :2] - p[0, :2]),
        heading_deg=abs(np.rad2deg(np.arctan2(np.sin(yaw), np.cos(yaw)))),
        pose_m=np.linalg.norm(
          root.inv().apply(real - real[0]) - desired.inv().apply(p - p[0]), axis=1
        ).mean(),
        joint_rmse_rad=np.mean((d.qpos[7:] - truth["joint_pos"][idx]) ** 2) ** 0.5,
        action=action,
        saturation=sat / (8 * 29),
        inference_ms=elapsed_ms,
      )
      for k, v in vals.items():
        trace[k].append(v)
      # Same height and tilt deviation termination on every controller.
      if (
        abs(real[0, 2] - p[0, 2]) > 0.25
        or abs(root.as_matrix()[2, 2] - desired.as_matrix()[2, 2]) > 0.8
      ):
        status = "fall"
        break
    return {
      **{k: np.array(v) for k, v in trace.items()},
      "initial": initial,
      "initial_velocity": initial_velocity,
    }, status


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  for arg in ("baseline", "actor", "reference", "out"):
    ap.add_argument(arg, type=Path)
  ap.add_argument("--rate", type=int, required=True)
  ap.add_argument("--preview-steps", type=int, required=True)
  ap.add_argument("--revision", choices=["none", "slow", "settle"], default="none")
  ap.add_argument("--bias-deg-s", type=float, default=0)
  a = ap.parse_args()
  torch.set_num_threads(1)
  arena = Arena(a.baseline)
  actor = Actor(a.actor)
  with np.load(a.reference, allow_pickle=True) as ref:
    if list(ref["joint_names"]) != arena.joints or ref["fps"].item() != 50:
      raise ValueError("Invalid reference order or fps")
    planner = Planner(ref, a.rate, a.preview_steps, a.revision)
  a.out.mkdir(parents=True, exist_ok=False)
  trace, status = arena.trial(
    actor, planner, np.array([0, 0, np.deg2rad(a.bias_deg_s)])
  )
  np.savez_compressed(a.out / "trace.npz", **trace)
  record = dict(
    task=actor.task,
    checkpoint=actor.identity,
    motion=a.reference.stem,
    planned=planner.length - 1,
    observed=len(trace["qpos"]),
    status=status,
    rate=a.rate,
    preview_steps=a.preview_steps,
    revision=a.revision,
    bias_deg_s=a.bias_deg_s,
    revision_tick=planner.switch,
  )
  (a.out / "trial.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
  main()
