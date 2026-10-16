"""Validate the captured artifacts and summarize pre-failure simulator states."""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
from pathlib import Path

import mujoco
import numpy as np


def main() -> None:
  evidence = Path(__file__).resolve().parent / "evidence"
  audit = json.loads((evidence / "audit.json").read_text())
  originals = {Path(a["path"]).name: a for a in audit["diagnostic_artifacts"]}
  dump_path = evidence / "nan_dump_20261016_045537.npz"
  model_name = "model_20261016_045537.mjb"
  model_bytes = gzip.decompress((evidence / (model_name + ".gz")).read_bytes())
  for name, data in (
    (dump_path.name, dump_path.read_bytes()),
    (model_name, model_bytes),
  ):
    assert len(data) == originals[name]["bytes"], name
    assert hashlib.sha256(data).hexdigest() == originals[name]["sha256"], name

  with tempfile.TemporaryDirectory(prefix="wbc-g1-se-evidence-") as tmp:
    model_path = Path(tmp) / model_name
    model_path.write_bytes(model_bytes)
    model = mujoco.MjModel.from_binary_path(str(model_path))

  # This exact dump is trusted only after matching its captured SHA-256 above.
  # mjlab stores its metadata as a pickled NumPy object array.
  with np.load(dump_path, allow_pickle=True) as dump:
    metadata = dump["_metadata"].item()
    assert metadata["model_file"] == model_name
    state_size = mujoco.mj_stateSize(model, metadata["state_spec"])
    assert state_size == metadata["state_size"]
    state_keys = sorted(
      (k for k in dump.files if k.startswith("states_step_")),
      key=lambda k: int(k.rsplit("_", 1)[1]),
    )
    assert len(state_keys) == metadata["buffer_size"]
    data = mujoco.MjData(model)
    states = []
    for key in state_keys:
      values = dump[key]
      assert values.shape == (metadata["num_envs_dumped"], state_size)
      for index, env_id in enumerate(metadata["dumped_env_ids"]):
        mujoco.mj_setState(model, data, values[index], metadata["state_spec"])
        states.append(
          {
            "step": int(key.rsplit("_", 1)[1]),
            "env_id": env_id,
            "all_saved_values_finite": bool(np.isfinite(values[index]).all()),
            "max_abs_qpos": float(np.abs(data.qpos).max()),
            "max_abs_qvel": float(np.abs(data.qvel).max()),
          }
        )
    result = {
      "artifact_checksums_match_server": True,
      "mujoco_version_used_for_inspection": mujoco.__version__,
      "model": {"nq": model.nq, "nv": model.nv, "na": model.na},
      "metadata": metadata,
      "states": states,
      "limitation": "The guard captures qpos/qvel before each physics step. "
      "These are pre-failure states, not the post-step tensor that triggered detection. "
      "They do not include the complete training state or all simulator inputs.",
    }
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
