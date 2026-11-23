"""Run paired planner conditions from a single partition, preserving individual traces."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from common import Actor, Arena
from planner import Planner


def worker(task):
  baseline, policy, rows, references, out, rate, horizon, revision = task
  torch.set_num_threads(1)
  arena, actor = Arena(baseline), Actor(policy)
  for row in rows:
    name = row["motion"]
    where = out / actor.task / f"{rate}hz-{horizon}steps-{revision}" / name
    where.parent.mkdir(parents=True, exist_ok=True)
    ref_path = references / row["split"] / "npz" / (name + ".npz")
    with ref_path.open("rb") as f:
      ref_hash = hashlib.file_digest(f, "sha256").hexdigest()
    identity = dict(
      task=actor.task,
      checkpoint=actor.identity,
      motion=name,
      family=row["family"],
      split=row["split"],
      source=row.get("source", name),
      rate=rate,
      preview_steps=horizon,
      revision=revision,
      canonical_sha256=row["sha256"],
      reference_sha256=ref_hash,
    )
    if where.exists():
      old = json.loads((where / "trial.json").read_text())
      with (where / "trace.npz").open("rb") as f:
        if (
          not all(old[k] == v for k, v in identity.items())
          or hashlib.file_digest(f, "sha256").hexdigest() != old["trace_sha256"]
        ):
          raise ValueError(f"Inconsistent saved trial {where}")
      continue
    with np.load(ref_path, allow_pickle=True) as ref:
      if list(ref["joint_names"]) != arena.joints or ref["fps"].item() != 50:
        raise ValueError(f"Invalid source order or rate for {ref_path}")
      planner = Planner(ref, rate, horizon, revision)
    trace, status = arena.trial(actor, planner, np.zeros(3))
    with tempfile.TemporaryDirectory(dir=where.parent) as tmp:
      staging = Path(tmp) / "trial"
      staging.mkdir()
      np.savez_compressed(staging / "trace.npz", **trace)
      with (staging / "trace.npz").open("rb") as f:
        identity["trace_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
      record = dict(
        **identity,
        planned=planner.length - 1,
        observed=len(trace["qpos"]),
        status=status,
        revision_tick=planner.switch,
      )
      (staging / "trial.json").write_text(json.dumps(record, indent=2) + "\n")
      os.rename(staging, where)
  return actor.task, rate, horizon, revision, len(rows)


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for k in ("baseline", "manifest", "references", "out"):
    p.add_argument(k, type=Path)
  p.add_argument("--actor", action="append", required=True, type=Path)
  p.add_argument("--split", required=True, choices=["validation", "test"])
  p.add_argument("--rates", nargs="+", type=int, required=True)
  p.add_argument("--horizons", nargs="+", type=int, required=True)
  p.add_argument(
    "--revisions", nargs="+", choices=["none", "slow", "settle"], required=True
  )
  p.add_argument("--workers", type=int, default=4)
  a = p.parse_args()
  with a.manifest.open() as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == a.split]
  if not rows:
    raise ValueError("No matching references")
  tasks = [
    (a.baseline, actor, rows, a.references, a.out, rate, h, rev)
    for actor, rate, h, rev in itertools.product(
      a.actor, a.rates, a.horizons, a.revisions
    )
  ]
  with ProcessPoolExecutor(a.workers) as pool:
    for done in pool.map(worker, tasks):
      print(done, flush=True)


if __name__ == "__main__":
  main()
