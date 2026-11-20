"""Compare common actor inputs and outputs on recorded native states."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import Actor, Arena, rot
from convert_reference import LinkKinematics


def replay(arena, actor, ref, states, expected, offsets, frames, previous):
  maximum = dict(observation=0.0, action=0.0, network=0.0)
  actor.start()
  for i, (q, v) in enumerate(states):
    actor.last = previous[i]
    indices = np.minimum(frames[i] + actor.offsets, len(ref["joint_pos"]) - 1)
    target = {
      k: v[indices]
      for k, v in ref.items()
      if k.startswith(("body_", "joint_")) and k != "joint_names"
    }
    inp = actor.inputs(target, arena.torso, rot(q[3:7]), v[3:6], q[7:], v[6:])
    if i not in offsets:
      continue
    saved_input, saved_action = expected[offsets[i]]
    native = actor.predict(saved_input[None])
    np.testing.assert_allclose(native, saved_action, rtol=1e-4, atol=1e-5)
    comparison = actor.predict(inp)
    maximum["observation"] = max(
      maximum["observation"], float(abs(inp[0] - saved_input).max())
    )
    maximum["action"] = max(
      maximum["action"], float(abs(comparison - saved_action).max())
    )
    maximum["network"] = max(
      maximum["network"], float(abs(native - saved_action).max())
    )
  return maximum


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  for arg in ("baseline", "actor", "motion", "output"):
    ap.add_argument(arg, type=Path)
  args = ap.parse_args()
  torch.set_num_threads(1)
  arena = Arena(args.baseline)
  actor = Actor(args.actor)
  if actor.sm is not None:
    z = np.load(args.actor / "parity.npz")
    with np.load(args.motion) as src:
      ref = LinkKinematics(arena.m).convert(src)
    n = len(z["action"])
    result = replay(
      arena,
      actor,
      ref,
      zip(z["qpos"], z["qvel"], strict=True),
      list(zip(z["input"], z["action"], strict=True)),
      {i: i for i in range(n)},
      z["frame"],
      np.vstack((np.zeros(29), z["action"][:-1])),
    )
    results = [dict(motion=args.motion.stem, samples=n, **result)]
  else:
    saved = args.actor / "evaluation"
    parity = np.load(saved / "export_parity.npz")
    initial = np.load(saved / "initial_states.npz")["state"]
    trials = json.loads((saved / "results.json").read_text())["trials"]
    expected = list(zip(parity["obs"], parity["actions"], strict=True))
    cursor = 0
    results = []
    for j, t in enumerate(trials):
      trace = np.load(saved / t["trace"])
      n = t["observed_steps"]
      ticks = [i for i in range(n) if i % 50 == 0 or i == t["steps"] - 1]
      offsets = {tick: idx + cursor for idx, tick in enumerate(ticks)}
      cursor += len(ticks)
      # One full trial per noncombat native clip exercises reset and time advance.
      if t["suite"] != "full" or t["name"].startswith("fight"):
        continue
      with np.load(args.motion / (t["name"] + ".npz")) as z:
        ref = {
          k: z[k] for k in z if k.startswith(("joint_", "body_")) and k != "joint_names"
        }
      q = np.vstack((initial[j, :36], trace["qpos"][:-1]))
      v = np.vstack((initial[j, 36:], trace["qvel"][:-1]))
      result = replay(
        arena,
        actor,
        ref,
        zip(q, v, strict=True),
        expected,
        offsets,
        np.arange(n),
        np.vstack((np.zeros(29), trace["action"][:-1])),
      )
      results.append(dict(motion=t["name"], samples=len(ticks), **result))
    if cursor != len(expected):
      raise ValueError("Parity trace and trial schedule disagree")
  args.output.write_text(json.dumps(results, indent=2) + "\n")
  print(results)
  if any(r["action"] > 1.0e-4 or r["observation"] > 1.0e-4 for r in results):
    raise ValueError("Native sensor/goal assembly mismatch; inspect recorded evidence")


if __name__ == "__main__":
  main()
