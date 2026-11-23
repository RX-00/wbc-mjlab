"""A 50 Hz reference clock with independent packet cadence and preview length."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def invalidate_past_end(packet, past):
  # The pose holds after the last sample. Its twist and joint velocity are zero.
  for name in ("joint_vel", "body_ang_vel_w", "body_lin_vel_w"):
    packet[name][past] = 0
  return packet


class Planner:
  def __init__(self, reference, rate, preview_steps, revision="none"):
    if rate not in (2, 5, 10, 50) or preview_steps < 0:
      raise ValueError("Expected a supported planner rate and nonnegative horizon")
    self.period = 50 // rate
    self.preview = preview_steps
    self.original = {
      k: reference[k]
      for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_ang_vel_w",
        "body_lin_vel_w",
      )
    }
    self.length = len(self.original["joint_pos"])
    # An integer second is a common publication instant for every tested rate.
    self.switch = min(max(1, self.length // 100) * 50, (self.length - 2) // 50 * 50)
    self.revised = self.original if revision == "none" else self.revise(revision)
    self.published_at = -1

  def revise(self, kind):
    if self.switch < 1:
      raise ValueError(
        "A revision trial needs a common publication tick before its endpoint"
      )
    # Stretch the remaining performance with a continuous change in speed.
    # Keep the same trial clock. The revised endpoint occurs earlier in the
    # source clip, as a receding-horizon planner may change its destination.
    t = np.arange(self.length, dtype=float)
    elapsed = np.maximum(0, (t - self.switch) / 50)
    if kind == "slow":
      speed = 1 - 0.25 * np.minimum(elapsed / 0.5, 1)
      u = t - 50 * 0.25 * np.where(elapsed < 0.5, elapsed**2, elapsed - 0.25)
    elif kind == "settle":
      # Validation diagnostic on upright clips selected for a planted endpoint.
      # Ramp speed to zero over 1 s, then hold the pose 25 source ticks ahead.
      if self.switch + 50 >= self.length:
        raise ValueError("A settle trial needs at least one second after revision")
      advance = np.minimum(elapsed, 1)
      speed = 1 - advance
      u = np.where(t <= self.switch, t, self.switch + 50 * (advance - 0.5 * advance**2))
    else:
      raise ValueError(kind)
    lo = u.astype(int)
    hi = np.minimum(lo + 1, self.length - 1)
    result = {}
    for name, value in self.original.items():
      if name == "body_quat_w":
        quat = np.empty_like(value)
        for j in range(value.shape[1]):
          quat[:, j] = Slerp(t, Rotation.from_quat(value[:, j, [1, 2, 3, 0]]))(
            u
          ).as_quat()[:, [3, 0, 1, 2]]
        result[name] = quat
      else:
        alpha = (u - lo).reshape((-1,) + (1,) * (value.ndim - 1))
        result[name] = value[lo] + alpha * (value[hi] - value[lo])
        if "vel" in name:
          result[name] *= speed.reshape(alpha.shape)
    return result

  def intent(self, tick):
    return self.revised if tick >= self.switch else self.original

  def update(self, tick):
    if tick % self.period == 0:
      source = self.intent(tick)
      indices = tick + np.arange(self.preview + 1)
      packet = {
        k: v[np.minimum(indices, self.length - 1)].copy() for k, v in source.items()
      }
      self.packet = invalidate_past_end(packet, indices >= self.length)
      self.published_at = tick
    if self.published_at != tick - tick % self.period:
      raise ValueError("Planner must receive each publication tick in sequence")

  def targets(self, tick, offsets):
    # Only the transmitted packet is visible at the consumer boundary.
    local = tick - self.published_at + np.asarray(offsets)
    if self.published_at < 0 or np.min(local) < 0:
      raise ValueError("Cannot read before the published plan")
    result = {
      k: v[np.minimum(local, self.preview)].copy() for k, v in self.packet.items()
    }
    return invalidate_past_end(result, local > self.preview)
