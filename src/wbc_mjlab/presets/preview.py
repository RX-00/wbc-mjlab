"""Match the GenCoord actor's sensor history and published motion fields."""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_conjugate, quat_mul

from wbc_mjlab.env.mdp.commands import MotionCommand

PREVIEW_COUNT = 10
PREVIEW_STRIDE = 5


def ref_motion_window(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  cmd: MotionCommand = env.command_manager.get_term(command_name)
  offsets = torch.arange(PREVIEW_COUNT, device=cmd.time_steps.device) * PREVIEW_STRIDE
  last = cmd.motion.segment_end_idx[cmd.trajectory_ids] - 1
  indices = torch.minimum(cmd.time_steps[:, None] + offsets, last[:, None])
  joint = torch.cat((cmd.motion.joint_pos[indices], cmd.motion.joint_vel[indices]), -1)
  pelvis = cmd.cfg.body_names.index("pelvis")
  orientations = cmd.motion.body_quat_w[indices, pelvis]
  # Ideal attitude during training. Deployment integrates the pelvis gyro
  # from the commanded starting alignment in the comparison adapter.
  inv = quat_conjugate(cmd.robot.data.root_link_quat_w)[:, None].expand_as(orientations)
  local = matrix_from_quat(quat_mul(inv, orientations))[..., :2]
  return torch.cat((joint.flatten(1), local.flatten(1)), -1)


def apply_preview(cfg):
  actor = cfg.observations["actor"]
  actor.history_length = None
  sensor_names = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions")
  actor.terms = {name: actor.terms[name] for name in sensor_names}
  for term in actor.terms.values():
    term.history_length = 10
  actor.terms["ref_motion_window"] = ObservationTermCfg(
    func=ref_motion_window, params={"command_name": "motion"}
  )
