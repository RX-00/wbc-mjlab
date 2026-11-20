"""Render saved evaluation states with a common camera and explicit outcomes."""

import argparse
import gzip
import json
import shutil
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from wbc_mjlab.scripts.evaluate import sha256


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--evaluation", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--clip", help="Exact clip name; default renders all full-clip trials"
  )
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  results = json.loads((args.evaluation / "results.json").read_text())
  model_path = args.evaluation / "model.mjb.gz"
  assert sha256(model_path) == results["model_sha256"]
  manifest = {
    "evaluation": str(args.evaluation),
    "task": results["task"],
    "actuator_model": results["actuator_model"],
    "videos": [],
  }
  try:
    with tempfile.TemporaryDirectory() as temporary:
      binary = Path(temporary) / "model.mjb"
      with gzip.open(model_path, "rb") as source, binary.open("wb") as destination:
        shutil.copyfileobj(source, destination)
      model = mujoco.MjModel.from_binary_path(str(binary))
      model.vis.global_.offwidth = 640
      model.vis.global_.offheight = 480
      data = mujoco.MjData(model)
      camera = mujoco.MjvCamera()
      camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
      camera.trackbodyid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "robot/torso_link"
      )
      assert camera.trackbodyid >= 0
      camera.distance, camera.azimuth, camera.elevation = 3.0, 135.0, -15.0
      fps = round(1 / results["protocol"]["control_period_s"])
      with (
        np.load(args.evaluation / "initial_states.npz") as initial,
        mujoco.Renderer(model, height=480, width=640) as renderer,
      ):
        for index, trial in enumerate(results["trials"]):
          if trial["suite"] != "full" or (args.clip and trial["name"] != args.clip):
            continue
          if trial["status"] == "not_started":
            manifest["videos"].append({"clip": trial["name"], "status": "not_started"})
            continue
          with np.load(args.evaluation / trial["trace"]) as trace:
            qpos = np.vstack(
              [initial["state"][index, : model.nq], trace["qpos"].reshape(-1, model.nq)]
            )
            qvel = np.vstack(
              [initial["state"][index, model.nq :], trace["qvel"].reshape(-1, model.nv)]
            )
          assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
          output = args.output / f"trial_{index:03d}.mp4"
          with imageio.get_writer(
            output, fps=fps, codec="libx264", quality=8
          ) as writer:
            for frame in range(len(qpos)):
              data.qpos[:], data.qvel[:] = qpos[frame], qvel[frame]
              mujoco.mj_forward(model, data)
              renderer.update_scene(data, camera=camera)
              image = Image.fromarray(renderer.render())
              draw = ImageDraw.Draw(image)
              draw.rectangle((0, 0, 640, 48), fill="black")
              draw.text(
                (8, 5),
                f"{results['task']} | {trial['name']} | {results['actuator_model']}",
                fill="white",
              )
              draw.text(
                (8, 24),
                f"t={frame / fps:.2f}s / {trial['steps'] / fps:.2f}s | {trial['status']}",
                fill="white",
              )
              writer.append_data(np.asarray(image))
          manifest["videos"].append(
            {
              "clip": trial["name"],
              "status": trial["status"],
              "video": output.name,
              "frames": len(qpos),
              "fps": fps,
            }
          )
      if not manifest["videos"]:
        raise ValueError("No matching full-clip trial")
    manifest["state"] = "completed"
  except Exception as error:
    manifest.update(state="failed", error=str(error))
    raise
  finally:
    (args.output / "videos.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
  main()
