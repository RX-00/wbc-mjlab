G1-SE physics failure: investigation handoff
===========================================

Evidence captured October 16, 2026, at 14:01 UTC (10:01 EDT). This is a
historical snapshot; query Slurm again before reporting current job status.
The requested work was to preserve evidence and push it to the personal fork.
No new fix, restart, cancellation, or campaign change was requested here.

Confirmed outcome
-----------------

The revised seed-42 campaign used training revision
``cbc0cf3b150732a8e242eeb2817e225d84019498``. G1 completed successfully, but
G1-SE job **12886** failed after its last logged PPO iteration **186124**.
The last saved G1-SE checkpoint is ``model_185000.pt``. Do not describe iteration
186125 as a completed update: detection occurred during subsequent collection.

The physics NaN guard reported::

   [NanGuard] Detected NaN/Inf at step 17868079

The cluster wrapper then failed the job with exit code 1::

   JOB FAILED: Nonfinite physics state; see persistent diagnostic dump

Slurm reports ``FAILED``, exit ``1:0``, after ``1-13:27:45`` on ``a2r-wolf``.
The end time was October 16 at 04:55:38 EDT (08:55:38 UTC). This was the
intentional numerical-failure stop, with a saved dump, rather than a successful
200,000-iteration result.

Dump metadata identifies environment **1476** of 2,048. It contains 32 states,
steps **17868047 through 17868078**, each with 71 values: 36 qpos and 35 qvel.
All saved values are finite. The maximum absolute qvel grows from about 53.3
in the first saved state to **29,163,178** in the last; the last maximum absolute
qpos is **145,816.40625**. See ``evidence/dump_summary.json`` for every state.
These measurements show rapidly diverging physics before detection. They do
not establish the initiating cause.

The distinction matters: mjlab's guard captures qpos/qvel **before** a physics
step, then checks qpos, qvel, qacc, qacc_warmstart, and sensordata **after** it.
The dump therefore does not identify which post-step tensor first became
nonfinite. It also does not include the full control, applied-wrench, contact,
per-environment randomized model, command, policy, optimizer, or RNG state.
Loading these states is possible; exact replay of the crash is not established.

Campaign context
----------------

All six tasks were submitted as fresh runs using the same source/image, sample motion dataset,
seed 42, 2,048 environments, 24 rollout steps, 200,000 iterations, save interval
5,000, ``njmax=1024``, and ``contact_sensor_maxmatch=128``. Physics NaN capture
used a 32-state buffer; both observation groups used ``nan_policy=error``.
Task-specific presets remain different by design. Exact launch commands and
common settings are in ``evidence/audit.json``; the failed run's actual saved
configuration is in ``evidence/params/``.

At evidence capture:

* G1, job 12885: ``COMPLETED``, exit ``0:0``; wrapper validated final checkpoint
  ``model_199999.pt`` and required exports.
* G1-SE, job 12886: ``FAILED`` as described above.
* Zest, job 12887: ``RUNNING``.
* Zest-SE, job 12888: ``RUNNING``.
* EE, job 12889, and EE-SE, job 12890: ``PENDING``.

Do not submit duplicate jobs. All six use the same deployed snapshot, including
the jobs still queued. Keep that snapshot immutable to preserve this comparison.

The earlier October 8 campaign had four failures and two completed EE tasks.
The original trigger was not reproduced. Revision ``cbc0cf3`` fixes two
independently reproduced reset defects: stale cached assistance observations
and stale derived kinematics after clip wrapping. It also skips assistance
calculation for terminal/zero-gain environments and adds fatal diagnostics.
The related regression tests are ``tests/test_motion_reset.py``,
``tests/test_assistive_wrench.py``, and ``tests/test_cluster_checks.py``. All 74
tests, six smoke jobs, and 1,000-iteration G1-SE/Zest-SE checks passed before
full training. Those checks did not establish long-run numerical stability;
this new G1-SE failure demonstrates a remaining issue. Do not claim the original
four crashes are fully explained or fixed.

Included artifacts and validation
---------------------------------

* ``evidence/audit.json``: timestamped Slurm output, campaign metadata and job IDs,
  completed/failed wrapper results, checkpoint paths, failure-log excerpt,
  original diagnostic sizes and SHA-256 hashes.
* ``evidence/failure.stdout.tail.txt`` and ``failure.stderr.txt``: failure context.
  The readable stdout excerpt removes ANSI styling and trailing whitespace;
  ``audit.json`` retains the original captured text.
* ``evidence/nan_dump_20261016_045537.npz``: original 20,160-byte state dump.
* ``evidence/model_20261016_045537.mjb.gz``: losslessly compressed matching model
  (about 34.8 MB compressed; 71,161,551 bytes decompressed).
* ``evidence/dump_summary.json``: metadata and per-state measurements, generated
  with the adjacent ``inspect_dump.py`` using MuJoCo 3.11.0.
* ``evidence/SHA256SUMS``: checksums of included evidence files.

From this directory, with the repository's locked dependencies installed::

   (cd evidence && sha256sum -c SHA256SUMS)
   python inspect_dump.py

The inspection script verifies the dump and decompressed model against the
original server hashes, loads the MJB, validates state dimensions and count,
and restores each saved state with ``mj_setState``. It uses a temporary directory
and prints JSON; it does not step physics or start training. The NPZ metadata
uses NumPy pickling, so the script verifies the original checksum before loading
that trusted metadata.

To use mjlab's existing dump viewer, decompress the model beside the NPZ under
the original basename first. The viewer expects the filename in metadata.
See installed ``mjlab/scripts/nan_viz.py`` (``viz-nan`` entry point). Visualization
is state playback, not proof that the failure can be reproduced.

Remote locations
----------------

Connect with ``ssh -p 2219 rxing@a2r-lab-server.dartmouth.edu``.

* Campaign: ``~/wbc-mjlab-runs/2026-10-14_cbc0cf3_fresh_seed42``.
* Original diagnostics, relative to campaign:
  ``diagnostics/train_Wbc-G1-SE_job12886/``.
* Failed run, relative to campaign:
  ``train/wbc_g1_se/2026-10-14_15-28-14_train_seed42_job12886/``.
  Latest checkpoint: ``model_185000.pt``. Checkpoints were not copied into this
  evidence bundle; avoid assuming their finiteness without inspecting them.
* Full console files, relative to campaign:
  ``console/wbc_train_Wbc-G1-SE-12886.out`` and ``.err``.
* Training source: ``~/wbc-mjlab-revisions/cbc0cf3``; no Git checkout there.
* Development clone: ``~/wbc-mjlab-fork``; local clone: ``/home/roy/wbc-mjlab``.
* Image: ``~/wbc-mjlab-images/2026-10-14_cbc0cf3/wbc-mjlab-cu128.sif``.
  SHA-256: ``c6bcb6916fa9bdc641b5d7126c3a5f5074143233260cc4029542e7894be62b29``.
* The older ``~/wbc-mjlab`` deployment belongs to the October 8 campaign.

Next investigation
------------------

1. Read this handoff, captured configuration, logs, and dump summary. Recheck
   Slurm and wrapper results for the other jobs; their states above are historical.
2. Inspect mjlab 1.6.0's ``utils/nan_guard.py`` and ``sim/sim.py`` to establish
   precisely what state is available and what additional inputs are missing.
3. Examine reference/action targets, assistance forces/torques, contacts, solver
   behavior, and randomized dynamics as hypotheses. The saved velocity growth
   alone does not distinguish these possible triggers.
4. If requested to implement a fix, reproduce the failure or a defensible causal
   mechanism, add meaningful regression coverage, and test in a separate snapshot.
   Do not sanitize invalid observations or silently reset bad worlds to claim
   training succeeded. Preserve the strict failure checks and campaign evidence.

Suggested skills
----------------

No specialist skill is required for SSH, Slurm, or simulator debugging. Use the
``handoff`` skill if another transfer of the investigation is requested. Follow
repository AGENTS instructions, preserve user changes, and keep verified facts
separate from hypotheses.
