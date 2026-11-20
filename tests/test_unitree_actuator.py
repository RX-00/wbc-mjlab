"""Exercise the dispatched motor law, independently of the envelope helper."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.actuator.actuator import ActuatorCmd

from wbc_mjlab.env.mdp.torque_envelope import TorqueSpeedEnvelopeTensors
from wbc_mjlab.robots.g1.actuators import G1_ACTUATOR_GROUPS, UnitreeActuator


def motor_for_test(cfg, shape):
  motor = object.__new__(UnitreeActuator)
  motor.cfg = cfg
  for name, value in (
    ("stiffness", cfg.stiffness),
    ("damping", cfg.damping),
    ("force_limit", float("inf")),
    ("_joint_vel", 0.0),
    ("_friction_static", cfg.fs),
    ("_friction_dynamic", cfg.fd),
    ("_activation_vel", cfg.va),
    ("motor_effort", 0.0),
  ):
    setattr(motor, name, torch.full(shape, value))
  motor._envelope = TorqueSpeedEnvelopeTensors(
    *(torch.full(shape, getattr(cfg, name)) for name in ("x1", "x2", "y1", "y2"))
  )
  return motor


@pytest.mark.parametrize("cfg", G1_ACTUATOR_GROUPS)
def test_compute_enforces_motor_envelope_before_passive_friction(cfg):
  speeds = [0, 0.001, cfg.x1, (cfg.x1 + cfg.x2) / 2, cfg.x2, cfg.x2 + 10]
  velocities = torch.tensor([speeds, [-v for v in speeds]])
  motor = motor_for_test(cfg, velocities.shape)
  for direction in (-1, 1):
    targets = torch.full_like(velocities, 1000 * direction)
    zero = torch.zeros_like(velocities)
    actual = motor.compute(ActuatorCmd(targets, zero, zero, zero, velocities))
    expected = torch.empty_like(actual)
    for env in range(2):
      for j in range(len(speeds)):
        v = velocities[env, j].item()
        # Independent scalar oracle, covering propulsion, braking and overspeed.
        peak = cfg.y1 if v * direction > 0 else cfg.y2
        taper = 1 if abs(v) <= cfg.x1 else max(0, (cfg.x2 - abs(v)) / (cfg.x2 - cfg.x1))
        friction = cfg.fs * math.tanh(v / cfg.va) + cfg.fd * v
        expected[env, j] = direction * peak * taper - friction
        assert friction * v >= 0
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


def test_runtime_gains_limits_and_environment_isolation():
  cfg = G1_ACTUATOR_GROUPS[2]
  motor = motor_for_test(cfg, (2, 1))
  zero = torch.zeros(2, 1)
  cmd = ActuatorCmd(torch.ones(2, 1), zero, zero, zero, zero)
  before = motor.compute(cmd).clone()
  motor.set_gains(torch.tensor([0]), kp=torch.tensor([1000.0]))
  motor.set_effort_limit(torch.tensor([0]), torch.tensor([2.0]))
  after = motor.compute(cmd)
  assert after[0].item() == 2.0
  torch.testing.assert_close(after[1], before[1])


def test_soft_torque_penalty_is_finite_active_motor_utilization():
  from wbc_mjlab.env.mdp.rewards import actuator_torque_soft_limit

  cfg = G1_ACTUATOR_GROUPS[2]
  motor = motor_for_test(cfg, (3, 1))
  zero = torch.zeros(3, 1)
  velocity = torch.tensor([[0.0], [50.0], [-50.0]])
  total = motor.compute(ActuatorCmd(zero + 1000, zero, zero, zero, velocity))
  env = SimpleNamespace(
    scene={"robot": SimpleNamespace(actuators=[motor])}, num_envs=3, device="cpu"
  )
  cost = actuator_torque_soft_limit(env)
  torch.testing.assert_close(cost, torch.tensor([0.05, 0.0, 0.0]))
  assert (total[1:] != 0).all()  # passive friction survives zero active capacity
  assert (motor.motor_effort[1:] == 0).all()


def test_evaluation_historical_model_reproduces_recorded_defect():
  from wbc_mjlab.scripts.evaluate import HistoricalActuator

  cfg = G1_ACTUATOR_GROUPS[2]
  motor = motor_for_test(cfg, (3, 1))
  historical = object.__new__(HistoricalActuator)
  historical.__dict__.update(motor.__dict__)
  zero = torch.zeros(3, 1)
  cmd = ActuatorCmd(
    torch.tensor([[10.0], [0.0], [0.0]]),
    zero,
    zero,
    zero,
    torch.tensor([[0.0], [50.0], [-50.0]]),
  )
  torch.testing.assert_close(
    historical.compute(cmd).flatten(),
    torch.tensor([142.5062, -48.9611, 48.9611]),
    atol=1e-4,
    rtol=1e-5,
  )
  torch.testing.assert_close(
    motor.compute(cmd).flatten(), torch.tensor([31.9, -3.6, 3.6]), atol=1e-4, rtol=1e-5
  )
