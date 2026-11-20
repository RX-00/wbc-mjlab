Corrected six-policy campaign: final results
============================================

As of November 10, all six fresh seed-42 policies completed 200,000 iterations,
shared evaluation and full-clip video rendering. All six final checkpoints,
saved configurations, exports and evaluation artifacts passed the recorded
integrity checks. No campaign jobs remain running or queued. This completes
the requested six-policy training/evaluation dataset; broader validation is
qualified by three unresolved GPU trajectory-comparison tests. No policy meets
all frozen tracking targets, and no historical crash cause is claimed resolved.

Numerical stability and artifact validation
------------------------------------------

All six successful attempts end at model_199999.pt, have finite actor/critic/
optimizer state (92 tensors each), and use the frozen v12 source, image and
common profile. Complete logs of these attempts contain no matched capacity
overflow, fatal NaN notification, traceback or native crash. The 240 evaluation
trials have zero numerical-failure outcomes. All 502,427 saved trace samples
are finite, frame-aligned and consistent with terminal states. Failures before
these successful attempts remain part of the campaign history below.

Both the training export and a fresh ONNX export pass the unchanged action
parity tolerances (atol=1e-5, rtol=1e-4) on 10,313 observations across six policies,
including independent local CPU rechecks. The largest absolute difference is
4.30e-6. All policies share protocol v3, ten training clips and their hashes,
trial schedules, metric body/joint sets, and bit-identical initial states and
controls. Assistance and automatic resets are disabled; actions are deterministic
policy means. GPU trajectory repeatability is limited, as described below.

Tracking quality
----------------

Errors are unweighted means of each trial's time-averaged error, including
samples through failure. Coverage is observed/planned control steps. A long
completed clip can dominate coverage, so coverage does not replace completion
rate. The protocol's ``fall`` outcome uses common anchor-height/orientation
error thresholds; it need not mean the robot is visibly on the ground.

.. csv-table:: Full-clip results
   :header: "Policy", "Job", "Completed", "Coverage", "Global body (m)", "Root-relative body (m)", "Joint RMSE (rad)"

   "G1", "20236", "8/10", "99.86%", "2.4649", "0.0433", "0.1426"
   "G1-SE", "19791", "8/10", "99.87%", "0.1305", "0.0475", "0.1355"
   "G1-Zest", "19792", "8/10", "98.95%", "2.2341", "0.0513", "0.1543"
   "G1-Zest-SE", "19794", "8/10", "99.86%", "0.1270", "0.0448", "0.1391"
   "G1-EE", "19795", "7/10", "98.33%", "0.1390", "0.0485", "0.1918"
   "G1-EE-SE", "19796", "7/10", "98.31%", "0.1247", "0.0475", "0.1797"

All six miss the frozen 90% full-clip completion target. G1-SE, Zest-SE, EE and
EE-SE meet the 0.15 m mean global-body-error target; G1 and Zest miss it by a
large margin. All six meet the 0.3 rad mean joint-RMSE target. Across the full
trials there are 46 completions and 14 falls. Low root-relative error does not
compensate for global drift: mean anchor drift is 2.4569 m for G1 and 2.2195 m
for Zest. These are single-seed observations, not statistically established
rankings between architectures.

Full-clip failures, in observed/planned control steps:

* G1: backflip 22/88, gainer_flip 95/141.
* G1-SE: backflip 35/88, gainer_flip 95/141.
* G1-Zest: backflip 35/88, dance1_subject1 5804/6573.
* G1-Zest-SE: backflip 23/88, gainer_flip 95/141.
* G1-EE: backflip 35/88, dance1_subject1 5355/6573, gainer_flip 97/141.
* G1-EE-SE: backflip 20/88, dance1_subject1 5352/6573, gainer_flip 105/141.

Quarter/half/three-quarter segment completion is 8/10, 8/10, 10/10 for every
policy except Zest, which achieves 9/10, 8/10, 10/10. Every difficult motion and
failed trial remains in the results. Per-trial JSON also reports orientation,
joint/body velocities, foot slip, joint-limit violations, motor saturation,
torque/power/work, contacts and penetration. Training-set nominal evaluation
with ideal simulator sensing does not establish generalization or deployment
performance; SE inputs here are not a trained state estimator.

Selected simulator and implemented repairs
-----------------------------------------

The common profile is 2.5 ms implicitfast, decimation 8 (50 Hz control), Newton
10 iterations / 20 line-search iterations, pyramidal cone, impratio 1,
njmax=1024 and contact_sensor_maxmatch=128. Training uses 2,048 environments,
24 rollout steps, fresh seed 42, checkpoint interval 5,000, original passive
friction and a 160 ms / 64-substep diagnostic ring. Model delays remain disabled.

The actuator now clips active PD torque through the existing Unitree envelope
in compute(), before subtracting unchanged passive friction. Regressions fail
on the old dispatch path and cover real initialized simulator effort. The
finite torque cost uses last-substep active torque divided by stationary peak
rating y2, excluding friction; its weight is -0.1. Reward-disabled pilots remain
separate evidence, not a conflated actuator-only comparison.

Matched short traces of SE/Zest/G1 at 2.5 ms have mean first-second joint RMSE
0.0209/0.0138/0.0149 rad against the 1 ms reference, compared with
0.0348/0.0341/0.0326 rad at 5 ms. Throughput is about 0.60–0.63 simulated
seconds/wall second at 2.5 ms, versus 0.96–1.01 at 5 ms and 0.30 at 1 ms.
Full-clip timestep results are mixed. Selection is a measured fidelity/cost
compromise, not proof that 1 ms is ground truth or that smaller steps fix the
historical failure. See ``evidence/profile-screen-v9.json`` and the two completed
10k pilot audits. Full implicit was not selected or used.

Preserved failures and validation limits
---------------------------------------

* G1-SE and Zest initially encountered Slurm NODE_FAIL. Their successful attempts
  restarted from scratch on October 30; interrupted checkpoints were preserved.
  The cluster failure cause is unknown. Only the new complete attempts qualify.
* G1 19793 failed with SIGSEGV after iteration 139320; a diagnostic continuation
  20229 also failed after 139744. The observed Python stack ends in a post-step
  finite-value reduction/host synchronization, without a captured physical NaN.
  Two GDB continuations (5k and 20k) completed without recurrence. Fresh G1 20236
  then completed under native stack/core capture with normal debugger exit.
  Instrumentation and successful retry do not establish a crash fix. All failed
  statuses, logs and checkpoint evidence remain preserved.
* CPU regression evidence: 88 passed, 20 GPU/slow cases deselected. Allocated
  GPU suite 19781: 105 passed, three failures in the Torch/ONNX terminal-qpos
  comparison at unchanged atol=rtol=1e-3 (corrected 2.5 ms, corrected 1 ms and
  historical-unbounded 5 ms). Action parity passed before those assertions.
  Same-backend repeats also vary; removing static friction only in a diagnostic
  reduces variability, implicating amplification by the explicit friction term.
  The exact backend source is unresolved. No test, tolerance or physical
  friction was weakened. The full GPU suite is not green, and terminal-state
  equivalence at that tolerance remains unverified.
* Historical numerical-crash causality remains unproven. Actuator, reward and
  timestep differ from that campaign; the successful runs cannot isolate a cause.
  Four historical checkpoints were evaluated under original and corrected
  physics (320 trials), separately from the fresh campaign. See
  ``historical-comparison.rst``.

Artifacts and reproducibility
-----------------------------

Machine-readable aggregate: ``evidence/six-policy-summary-20261110.json``.
Per-policy results and wrapper statuses: ``evidence/final-v12-results/``.
Full local checkpoints, configurations, exports, initial states, traces and
videos: ``logs/stability-final-v12/completed/<task>-<job>/`` at the repository
root. The authoritative server campaign is
``/home/rxing/wbc-mjlab-runs/2026-10-26_stability/final-v12-2p5ms``.
Source/image/version/data hashes remain in ``final-campaign-candidate.json``.

All 60 full videos have verified hashes, decoded frame counts (467,654 total),
50 Hz rate and 640x480 dimensions. First/middle/last frames of every clip were
visually inspected, including failures. This is sampled review, not continuous
viewing; the common following camera cannot independently validate global drift.
G1-SE/Zest audited videos are in full-videos-cpu; the other four use
full-videos-gpu. Their contact sheets are alongside those directories.

The final three policies are covered by
``evidence/final-three-remote-audit-20261110.json`` and
``evidence/final-three-results-audit-20261110.json``; earlier three retain their
November 2–5 audits. Source and image were rechecked November 10 in
``evidence/final-source-image-audit-20261110.json``. Local Python/runtime tests
match the frozen source bytewise or by identical AST where formatting differs;
local launchers additionally enable Python fault handling. All previous sealed
evidence hashes were verified before adding the new audit files.
