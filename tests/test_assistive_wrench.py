"""Inactive and terminal worlds must not contribute invalid wrench values."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from wbc_mjlab.env.mdp.assistive_wrench import (
  AssistiveWrenchEvent,
  AssistiveWrenchNominal,
)


def test_wrench_excludes_inactive_and_terminal_worlds():
  zeros = torch.zeros(3, 3)
  quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
  command = SimpleNamespace(
    episode_assist_gain=torch.tensor([0.0, 0.5, 0.5]),
    assist_force_w=zeros.clone(),
    assist_torque_w=zeros.clone(),
    anchor_pos_w=zeros.clone(),
    anchor_quat_w=quats.clone(),
    anchor_lin_vel_w=zeros.clone(),
    anchor_ang_vel_w=zeros.clone(),
    anchor_lin_acc_w=zeros.clone(),
    anchor_ang_acc_w=zeros.clone(),
  )
  velocity = zeros.clone()
  velocity[:2] = float("nan")
  velocity[2] = torch.tensor([1.0, 2.0, 3.0])
  asset = SimpleNamespace(
    data=SimpleNamespace(
      body_link_pos_w=zeros[:, None],
      body_link_quat_w=quats[:, None],
      body_link_lin_vel_w=zeros[:, None],
      body_link_ang_vel_w=velocity[:, None],
      root_link_pos_w=zeros,
      data=SimpleNamespace(subtree_com=zeros[:, None]),
    ),
    indexing=SimpleNamespace(root_body_id=0),
    write_external_wrench_to_sim=Mock(),
  )
  env = SimpleNamespace(
    scene={"robot": asset},
    num_envs=3,
    device="cpu",
    reset_buf=torch.tensor([False, True, False]),
    command_manager=SimpleNamespace(get_term=lambda name: command),
  )
  event = AssistiveWrenchEvent(None, env)
  event._nominal = AssistiveWrenchNominal(
    mass=10.0,
    inertia=torch.diag(torch.tensor([1.0, 2.0, 3.0])),
    gravity=torch.tensor([0.0, 0.0, -9.81]),
  )
  event._anchor_body_index = 0
  event(env, None, "motion")
  assert torch.equal(command.assist_force_w[:2], torch.zeros(2, 3))
  assert torch.equal(command.assist_torque_w[:2], torch.zeros(2, 3))
  # -10 I omega + omega x I omega, then beta=0.5.
  torch.testing.assert_close(
    command.assist_torque_w[2], torch.tensor([-2.0, -23.0, -44.0])
  )
  torch.testing.assert_close(command.assist_force_w[2], torch.tensor([0.0, 0.0, 4.905]))
  forces, torques = asset.write_external_wrench_to_sim.call_args.args
  torch.testing.assert_close(forces[:, 0], command.assist_force_w)
  torch.testing.assert_close(torques[:, 0], command.assist_torque_w)
  # A corrupt active world must remain visible to strict observation diagnostics.
  velocity[2] = float("nan")
  event(env, None, "motion")
  assert not torch.isfinite(command.assist_torque_w[2]).all()
