"""Packet-boundary and short-clip regressions for the common reference clock."""

import numpy as np
import pytest
from scripts.comparison.planner import Planner


def reference(n):
  t = np.arange(n, dtype=float)
  q = np.zeros((n, 1, 4))
  q[:, :, 0] = 1
  return dict(
    joint_pos=t[:, None],
    joint_vel=np.full((n, 1), 50.0),
    body_pos_w=np.broadcast_to(t[:, None, None], (n, 1, 3)),
    body_quat_w=q,
    body_ang_vel_w=np.zeros((n, 1, 3)),
    body_lin_vel_w=np.full((n, 1, 3), 50.0),
  )


@pytest.mark.parametrize("rate", [2, 5, 10])
def test_pre_revision_packet_cannot_see_future_stop(rate):
  p = Planner(reference(201), rate, 45, "settle")
  for t in range(p.switch):
    p.update(t)
  tick = p.switch - 1
  from_packet = p.targets(tick, [0, 5])["joint_pos"][:, 0]
  np.testing.assert_array_equal(from_packet, [tick, tick + 5])
  p.update(p.switch)
  assert p.targets(p.switch, [5])["joint_pos"].item() < p.switch + 5
  np.testing.assert_allclose(p.revised["joint_pos"][p.switch + 50 :], p.switch + 25)
  np.testing.assert_array_equal(p.revised["joint_vel"][p.switch + 50 :], 0)


def test_expiry_and_short_clip_revision():
  p = Planner(reference(90), 2, 5, "slow")
  assert p.switch == 50
  p.update(0)
  packet = p.targets(7, [0, 5, 45])
  np.testing.assert_array_equal(packet["joint_pos"], 5)
  np.testing.assert_array_equal(packet["joint_vel"], 0)
