"""Separate packet truncation, gyro integration and native/common rollout differences."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import Actor, Arena, rot
from convert_reference import LinkKinematics
from planner import Planner


class DiagnosticActor(Actor):
  def inputs(self, targets, torso, attitude, gyro, position, velocity):
    # The ideal-attitude condition is a diagnostic upper bound, excluded from ranking.
    if self.ideal:
      attitude = rot(self.arena.d.qpos[3:7])
    return super().inputs(targets, torso, attitude, gyro, position, velocity)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for k in ("baseline", "sm_actor", "sm_motion", "wbc_motions", "output"):
    p.add_argument(k, type=Path)
  a = p.parse_args()
  torch.set_num_threads(1)
  arena = Arena(a.baseline)
  rows = []
  for weights, motion in (
    (a.sm_actor, a.sm_motion),
    (a.baseline, a.wbc_motions / "dance.npz"),
  ):
    actor = DiagnosticActor(weights)
    actor.arena = arena
    with np.load(motion, allow_pickle=True) as f:
      if actor.sm is not None:
        ref = LinkKinematics(arena.m).convert(f)
        with np.load(weights / "parity.npz") as z:
          q0, v0 = z["qpos"][0], z["qvel"][0]
          native_q, native_a = z["qpos"][1:], z["action"][:-1]
      else:
        ref = {
          k: f[k] for k in f if k.startswith(("body_", "joint_")) and k != "joint_names"
        }
        e = weights / "evaluation"
        trials = json.loads((e / "results.json").read_text())["trials"]
        j, t = next(
          (j, t)
          for j, t in enumerate(trials)
          if t["suite"] == "full" and t["name"] == motion.stem
        )
        z = np.load(e / "initial_states.npz")["state"][j]
        q0, v0 = z[:36], z[36:]
        with np.load(e / t["trace"]) as z:
          native_q, native_a = z["qpos"], z["action"]
    for rate, preview, ideal, bias in (
      (50, 45, True, 0),
      (50, 45, False, 0),
      (2, 45, False, 0),
      (2, 5, False, 0),
      (2, 45, False, 0.5),
    ):
      actor.ideal = ideal
      trace, status = arena.trial(
        actor, Planner(ref, rate, preview), np.array([0, 0, np.deg2rad(bias)])
      )
      n = min(len(native_a), len(trace["action"]), 100)
      rows.append(
        dict(
          task=actor.task,
          motion=motion.stem,
          rate=rate,
          preview_steps=preview,
          ideal_attitude=ideal,
          gyro_bias_deg_s=bias,
          initial_qpos_max=float(abs(q0 - trace["initial"]).max()),
          initial_qvel_max=float(abs(v0 - trace["initial_velocity"]).max()),
          action_first_max=float(abs(native_a[0] - trace["action"][0]).max()),
          paired_steps=n,
          native_common_joint_rmse_rad=float(
            np.mean((trace["qpos"][:n, 7:] - native_q[:n, 7:]) ** 2) ** 0.5
          ),
          native_common_root_xy_m=float(
            np.linalg.norm(trace["qpos"][:n, :2] - native_q[:n, :2], axis=1).mean()
          ),
          common_status=status,
          observed=len(trace["action"]),
          path_mean_m=float(trace["path_m"].mean()),
          heading_mean_deg=float(trace["heading_deg"].mean()),
          pose_mean_m=float(trace["pose_m"].mean()),
        )
      )
  a.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
  main()
