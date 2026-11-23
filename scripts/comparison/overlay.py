"""Render paired terminal-state traces at 50 Hz with a common reference ghost."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from convert_reference import load_model
from PIL import Image, ImageDraw, ImageFont


def load_trial(folder):
  record = json.loads((folder / "trial.json").read_text())
  with (folder / "trace.npz").open("rb") as f:
    if hashlib.file_digest(f, "sha256").hexdigest() != record["trace_sha256"]:
      raise ValueError(f"Trace digest mismatch in {folder}")
  with np.load(folder / "trace.npz") as z:
    trace = dict(z)
  if len(trace["qpos"]) != record["observed"]:
    raise ValueError("Recorded timebase mismatch")
  return record, trace


def render(model, folders, labels, output, credit=None):
  trials = [load_trial(p) for p in folders]
  r, z = max(trials, key=lambda trial: trial[0]["observed"])
  for other, t in trials:
    for key in (
      "motion",
      "rate",
      "preview_steps",
      "revision",
      "planned",
      "reference_sha256",
    ):
      if other[key] != r[key]:
        raise ValueError(f"Unpaired {key}")
    np.testing.assert_array_equal(t["ref_qpos"], z["ref_qpos"][: other["observed"]])
    np.testing.assert_array_equal(t["initial"], z["initial"])
    np.testing.assert_array_equal(t["initial_velocity"], z["initial_velocity"])
  m = load_model(model)
  ref, actual = mujoco.MjData(m), mujoco.MjData(m)
  option, perturb = mujoco.MjvOption(), mujoco.MjvPerturb()
  camera = mujoco.MjvCamera()
  camera.azimuth, camera.elevation = 110, -13
  width, height = 480, 360
  font = ImageFont.truetype("DejaVuSans.ttf", 14)
  panel_height = height + 64 + (22 if credit else 0)
  output.parent.mkdir(parents=True, exist_ok=True)
  # Camera moves with the reference; both panels use identical framing and zoom.
  max_gap = max(
    np.linalg.norm(t["qpos"][:, :2] - t["ref_qpos"][:, :2], axis=1).max()
    for _, t in trials
  )
  camera.distance = max(3.1, 2.2 + 1.8 * max_gap)
  with (
    mujoco.Renderer(m, width=width, height=height) as renderer,
    imageio.get_writer(
      output,
      fps=50,
      codec="libx264",
      quality=8,
      macro_block_size=1,
      ffmpeg_log_level="error",
    ) as writer,
  ):
    for i in range(r["observed"]):
      ref.qpos[:] = z["ref_qpos"][i]
      mujoco.mj_forward(m, ref)
      camera.lookat[:] = [*ref.qpos[:2], 0.68]
      canvas = Image.new("RGB", (width * len(trials), panel_height + 36), "#162026")
      draw = ImageDraw.Draw(canvas)
      marker = ""
      if i + 1 >= r["revision_tick"]:
        marker = {"slow": "  |  Revised speed", "settle": "  |  Settle to stand"}.get(
          r["revision"], ""
        )
      title = f"{r['motion']}     {(i + 1) / 50:.2f} s     {r['rate']} Hz / {r['preview_steps'] / 50:.2f} s preview{marker}"
      draw.text((9, 9), title, font=font, fill="white")
      for j, ((other, t), label) in enumerate(zip(trials, labels, strict=True)):
        n = min(i, other["observed"] - 1)
        actual.qpos[:] = t["qpos"][n]
        mujoco.mj_forward(m, actual)
        renderer.update_scene(actual, camera=camera)
        scene = renderer.scene
        count = scene.ngeom
        mujoco.mjv_addGeoms(
          m, ref, option, perturb, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene
        )
        for idx in range(count, scene.ngeom):
          g = scene.geoms[idx]
          g.rgba[:] = [0.06, 0.84, 0.97, 0.40]
          g.matid = -1
          g.emission = 0.45
        canvas.paste(Image.fromarray(renderer.render()), (j * width, 36))
        status = (
          "ended: fall"
          if i >= other["observed"] - 1 and other["status"] == "fall"
          else "tracking"
        )
        if status == "tracking" and i + 1 == other["planned"]:
          status = "finished"
        x = j * width + 9
        draw.text((x, height + 39), f"{label}  |  {status}", font=font, fill="white")
        draw.text(
          (x, height + 58),
          f"Path {t['path_m'][n]:.2f} m    Heading {t['heading_deg'][n]:.0f}°    Pose {t['pose_m'][n]:.2f} m",
          font=font,
          fill="white",
        )
        draw.text((x, height + 78), "Cyan = commanded pose", font=font, fill="#38d9ed")
      if credit:
        draw.text((9, height + 99), credit, font=font, fill="white")
      writer.append_data(np.asarray(canvas))


if __name__ == "__main__":
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("model", type=Path)
  p.add_argument("output", type=Path)
  p.add_argument("--trial", action="append", required=True, type=Path)
  p.add_argument("--label", action="append", required=True)
  p.add_argument("--credit")
  a = p.parse_args()
  render(a.model, a.trial, a.label, a.output, a.credit)
