"""Expand canonical G1 pelvis/joint references into v12 body-link trajectories."""

import argparse
import csv
import gzip
import hashlib
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from wbc_mjlab.motion.motion_export_bundle import save_motion_npz


def load_model(path):
  with tempfile.NamedTemporaryFile(suffix=".mjb") as f:
    f.write(gzip.decompress(path.read_bytes()))
    f.flush()
    return mujoco.MjModel.from_binary_path(f.name)


class LinkKinematics:
  def __init__(self, model):
    self.model = model
    self.data = mujoco.MjData(model)
    self.joints = [
      model.joint(i).name.removeprefix("robot/") for i in range(1, model.njnt)
    ]
    self.bodies = [
      i for i in range(model.nbody) if model.body(i).name.startswith("robot/")
    ]
    if model.nq != 36 or model.body(self.bodies[0]).name != "robot/pelvis":
      raise ValueError("Expected the floating-base 29-joint v12 G1")

  def convert(self, src):
    if float(src["fps"].item()) != 50 or list(src["joint_names"]) != self.joints:
      raise ValueError("Reference rate or joint order differs from the native model")
    n = len(src["joint_pos"])
    rot = Rotation.from_quat(src["root_quat_w"][:, [1, 2, 3, 0]])
    pos = np.column_stack((src["root_pos_w"], src["root_quat_w"], src["joint_pos"]))
    vel = np.column_stack(
      (src["root_lin_vel_w"], rot.inv().apply(src["root_ang_vel_w"]), src["joint_vel"])
    )
    if not np.isfinite(pos).all() or not np.isfinite(vel).all():
      raise ValueError("Nonfinite reference")
    out = {k: src[k] for k in ("fps", "joint_pos", "joint_vel")}
    widths = {
      "body_pos_w": 3,
      "body_quat_w": 4,
      "body_ang_vel_w": 3,
      "body_lin_vel_w": 3,
    }
    out.update(
      {k: np.empty((n, len(self.bodies), w), np.float32) for k, w in widths.items()}
    )
    m, d, ids = self.model, self.data, self.bodies
    for i in range(n):
      d.qpos[:], d.qvel[:] = pos[i], vel[i]
      mujoco.mj_kinematics(m, d)
      mujoco.mj_comPos(m, d)
      mujoco.mj_comVel(m, d)
      omega = d.cvel[ids, :3]
      out["body_pos_w"][i] = d.xpos[ids]
      out["body_quat_w"][i] = d.xquat[ids]
      out["body_ang_vel_w"][i] = omega
      out["body_lin_vel_w"][i] = d.cvel[ids, 3:] + np.cross(
        omega, d.xpos[ids] - d.subtree_com[ids[0]]
      )
    for suffix in ("pos_w", "lin_vel_w", "ang_vel_w"):
      np.testing.assert_allclose(
        out["body_" + suffix][:, 0], src["root_" + suffix], atol=2e-5
      )
    return out


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("manifest", type=Path)
  ap.add_argument("source", type=Path)
  ap.add_argument("model", type=Path)
  ap.add_argument("output", type=Path)
  args = ap.parse_args()
  kin = LinkKinematics(load_model(args.model))
  with args.manifest.open() as f:
    rows = list(csv.DictReader(f))
  args.output.mkdir(parents=True, exist_ok=False)
  for row in rows:
    path = args.source / (row["motion"] + ".npz")
    with path.open("rb") as f:
      if hashlib.file_digest(f, "sha256").hexdigest() != row["sha256"]:
        raise ValueError(f"Motion hash mismatch for {path}")
    with np.load(path) as src:
      out = kin.convert(src)
    save_motion_npz(
      output_path=args.output / row["split"] / "npz" / path.name,
      log=out,
      robot_id="g1",
      joint_names=kin.joints,
      robot_body_names=[kin.model.body(i).name for i in kin.bodies],
    )
  print(f"Wrote {len(rows)} references under {args.output}")


if __name__ == "__main__":
  main()
