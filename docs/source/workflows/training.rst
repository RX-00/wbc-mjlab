.. _training:

Train a policy
==============

Launch a training run, follow it, and continue it later. Runs are long — WBC tasks default
to 200k PPO iterations and are meant to be stopped and resumed, not restarted.

.. admonition:: You need
   :class: tip

   A converted motion library (:doc:`../data`) and a task (:doc:`../tasks/index`).

Launch a run
------------

.. tab-set::
   :sync-group: ui

   .. tab-item:: Studio
      :sync: studio

      *Train*: the preset and dataset come from the previous steps. Set the environments and
      iterations, a run name and the logger, then **Launch**. The job runs detached — it
      survives closing the browser or restarting the Studio.

   .. tab-item:: Command line
      :sync: cli

      .. code-block:: bash

         uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan \
           --env.scene.num-envs 8192 --agent.max-iterations 30000 --agent.run-name first

``wbc-mjlab-train`` wraps mjlab's training script: it adds ``--task`` and the motion-data
options, and passes every other option through (``--agent.*``, ``--env.*``, ``--gpu-ids`` …).
All options: :ref:`cli-train` · ``uv run wbc-mjlab-train Wbc-G1 --help`` shows the task's
defaults.

Follow progress
---------------

- **Studio** — *Train → Jobs*: progress and ETA, curves (reward, tracking errors, per-term
  rewards, sampling, losses), the log, and a TensorBoard button.
- **TensorBoard** — ``tensorboard --logdir logs/rsl_rl``; ``--agent.logger wandb`` for W&B.
- **Watch the policy** — load any checkpoint mid-training in the Studio's *Play* step or with
  ``wbc-mjlab-play`` (:doc:`../visualization`).

Each run writes ``logs/rsl_rl/<experiment>/<timestamp>_<run_name>/``: checkpoints, TensorBoard
events and ``params/`` (configs, ``policy.onnx``, ``config.yaml``, sampling state) —
:ref:`reference-configuration`.

Train a Studio preset
---------------------

Presets edited in the Studio train like a task — from the Studio's *Train* step, or from the
command line:

.. code-block:: bash

   uv run python -m wbc_mjlab.studio.train_preset --preset my_preset --dataset samples \
     --env.scene.num-envs 4096 --agent.max-iterations 30000

All ``wbc-mjlab-train`` flags pass through. Single GPU only: ``torchrunx`` workers would not
see the preset task. See :doc:`../studio`.

Resume a run
------------

.. code-block:: bash

   # first launch
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --cache-motion-bundle

   # continue (same task + dataset)
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --cache-motion-bundle \
     --agent.resume True

What ``--agent.resume True`` does
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **New run folder** — each launch creates a fresh timestamped directory under
   ``logs/rsl_rl/<experiment_name>/`` (logs/checkpoints for the continued session).
2. **Load latest checkpoint** — mjlab searches ``logs/rsl_rl/<experiment_name>/`` for
   the most recent run matching ``--agent.load-run`` (default ``.*``) and checkpoint
   matching ``--agent.load-checkpoint`` (default ``model_.*.pt``).
3. **Restore training state** — policy (+ optimizer) weights via RSL-RL ``runner.load``.
4. **Restore RSI curriculum** — wbc-mjlab's runner also loads ``rsi_bin_stats.npz`` from
   the checkpoint run's ``params/`` when ``persist_failure_levels=True`` (WBC/Zest
   presets), so adaptive bin failure EMA continues where you left off.

Pick a specific prior run:

.. code-block:: bash

   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan \
     --agent.resume True \
     --agent.load-run 2026-07-03_14-30-00 \
     --agent.load-checkpoint model_5000.pt

Use the **same ``--task`` and motion data** as the original run so env dimensions and
motion library match the checkpoint.

Train on several GPUs
---------------------

mjlab launches **one process per GPU** via ``torchrunx`` when more than one GPU is
selected. Each worker gets a subset of parallel envs on its device; gradients are
synchronized by RSL-RL.

.. code-block:: bash

   # four GPUs on one machine
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --gpu-ids 0 1 2 3

   # all visible GPUs (respects CUDA_VISIBLE_DEVICES)
   uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --gpu-ids all

   # restrict visible devices first
   CUDA_VISIBLE_DEVICES=2,3 uv run wbc-mjlab-train --task Wbc-G1 --dataset lafan --gpu-ids 0 1

``--gpu-ids`` indices are **into** ``CUDA_VISIBLE_DEVICES``, not necessarily physical
GPU ids. Single-GPU (default ``--gpu-ids 0``) runs in-process without torchrunx.

Worker logs go to ``<run_dir>/torchrunx/`` unless overridden with ``--torchrunx-log-dir``.

Slurm and Apptainer
-------------------

``scripts/cluster/`` provides an Apptainer build and separate Slurm jobs for
the six WBC/Zest-family G1 tasks. The launcher defaults to the A2R cluster's
``debug`` partition, one RTX 5090, eight CPUs, 16 GiB RAM, and exclusion of
``a2r-main``. The batch template pins jobs to ``a2r-wolf`` by default. Pending
jobs can use either wolf or fox by clearing Slurm's ``ReqNodeList`` and setting
``ExcNodeList=a2r-main,a2r-tiger`` for the current four-node debug partition.
Check the current partition membership before applying those exclusions.
Update dependencies as well when enabling concurrent training; retain afterok
gates for prerequisite qualification runs. These scheduler updates preserve the
immutable source snapshot and training configuration. Adapt the resource
directives for another cluster.

Build on a machine with Apptainer fakeroot support and sufficient local scratch
space. First convert the bundled sample motions. The build script snapshots the
tracked checkout, local scripts, and converted samples, and installs the frozen
CUDA dependencies from ``uv.lock`` into a Python 3.12 image:

.. code-block:: bash

   bash scripts/cluster/build_image.sh

The printed build directory contains ``context/`` (the source/data snapshot),
``wbc-mjlab-cu128.sif``, and ``IMAGE.sha256``. Copy ``context/`` to the shared
cluster checkout and the image plus its checksum to one shared image directory.
Keep the checkout fixed throughout the campaign. Jobs validate the snapshot and
image checksums, preserve Slurm's allocated GPU visibility, and use a node-local
Warp cache.

Submit short validation runs, inspect all six job results, then submit training
into the same campaign root:

.. code-block:: bash

   bash scripts/cluster/submit_six.sh /shared/images/wbc-mjlab-cu128.sif /shared/runs/campaign smoke
   bash scripts/cluster/submit_six.sh /shared/images/wbc-mjlab-cu128.sif /shared/runs/campaign stability
   bash scripts/cluster/submit_six.sh /shared/images/wbc-mjlab-cu128.sif /shared/runs/campaign train

The jobs run in order: G1-SE, Zest, G1, Zest-SE, EE, EE-SE. Each depends on the
previous job succeeding, so a failure blocks the remaining jobs for investigation.
Additional arguments after the phase are forwarded to ``run_checked.py``;
for example, append ``--physics-dt 0.0025 --evaluate`` for a qualified 2.5 ms
profile with final evaluation. Choose the profile from validation results before
starting the full campaign. Every full
run starts fresh with seed 42, 2,048 environments, 24 rollout steps, 200,000 PPO
iterations, and checkpoint interval 5,000. The sample motions are loaded from the
snapshot. All tasks use ``njmax=1024`` to provide more constraint capacity than
the G1 default of 250, and ``contact_sensor_maxmatch=128`` (the original sample
campaign reported a required capacity of 65). The runner rejects every reported
overflow, including broadphase collision and contact-sensor overflow,
propagates training failures, and verifies the final checkpoint, saved common
settings, and exported artifacts. Smoke results are separate and are never
loaded into full training.

The optional ``stability`` phase runs 1,000 fresh PPO iterations per task. Run
smoke and stability checks to completion before submitting full training; the
launcher does not automatically gate one phase on another.

Cluster jobs enable per-term observation checks with ``nan_policy=error`` for
both actor and critic, and a 32-state physics NaN guard at the default 5 ms timestep. A physics NaN/Inf dump
also fails the job, even if an environment reset would otherwise hide it.
Dumps and model files persist under
``diagnostics/<phase>_<task>_job<id>/`` inside the campaign. The JSON result records
this directory. These checks stop on invalid values; they do not sanitize them.

The WBC runner also saves ``physical_failure_*.pt`` and JSON metadata before
the standard fatal notification: offending array/indices/worlds, recent state,
controls, applied wrenches, warmstarts, current randomized model parameters,
actions and targets. ``capture_capacity.json`` records ring memory and duration.
Checks cover raw actions, processed actions, physics inputs/outputs, sensing and
raw rewards before the learner consumes them. These are investigation bundles,
not complete closed-loop replay or optimizer snapshots.

``run_checked.py --physics-dt`` supports 0.005, 0.0025 and 0.001 s. It holds the
control period at 20 ms, the self-contact history at 20 ms, and diagnostic history
at 160 ms. Actuator command delays remain disabled. Reference advancement,
observation histories, assistance updates and PPO rollout lengths remain tied to
the unchanged control period. The launcher validates these saved settings.
Use ``--torque-penalty disabled`` for an actuator-only ablation; the default
``corrected`` uses active motor torque divided by the finite stationary peak
rating, excluding passive friction. Report these experiments separately.

Pass ``--evaluate`` to ``run_checked.py`` (or after the six positional arguments
to ``train.sbatch``) to evaluate the final checkpoint before the job is marked
complete. This checks execution and export parity; inspect the tracking scores
separately. The launcher does not turn poor completion rates into successful
tracking claims.

``jobs_smoke.tsv``, ``jobs_stability.tsv``, and ``jobs_train.tsv`` record task/job IDs. Slurm output is in
``console/``, training outputs are in ``smoke/``, ``stability/``, or ``train/``, and finished jobs
write JSON results in ``status/``. Missing JSON may indicate that Slurm killed a
job before the wrapper finished; always also check ``sacct``. A normal fresh
200,000-iteration run ends at zero-based checkpoint ``model_199999.pt``.

Both batch launchers enable ``PYTHONFAULTHANDLER=1`` inside the container so
native fatal signals can emit Python thread stacks to the job logs. For an
older immutable launcher, submit with
``--export=ALL,APPTAINERENV_PYTHONFAULTHANDLER=1``; verify the container and its
Python child see ``faulthandler.is_enabled()`` before relying on it. A training
subprocess exit of ``-11`` denotes SIGSEGV, not a diagnosed physics NaN. Preserve
the failed status, logs and checkpoints and investigate before retrying.
An ``afterok`` dependency on a failed job cannot be satisfied: point dependent
jobs at the explicitly submitted replacement, retaining their success gates.
Video replacements must also reference the replacement evaluation directory.

Tips
----

- **Fewer envs for debugging:** ``--env.scene.num-envs 512``
- **Extend horizon:** ``--agent.max-iterations 300000`` (works with resume)
- **Faster restarts on big libraries:** ``--cache-motion-bundle``

.. seealso::

   - :doc:`deploy` — export the trained policy for a robot
   - :doc:`../concepts/adaptive_sampling` — what the sampling curves show
   - :ref:`cli-train` — every option

Headless tracking evaluation
---------------------------

For the six G1 policies, run a fixed nominal protocol in a new output directory:

.. code-block:: bash

   uv run python -m wbc_mjlab.scripts.evaluate --task Wbc-G1-SE \
     --checkpoint /path/to/model_999.pt --motion-file data/g1/samples \
     --output /path/to/evaluation

Protocol v3 evaluates all full clips plus quarter-spaced segments of up to 5 s.
It uses deterministic policy means, explicit reference initial states, nominal
model parameters, ideal simulator sensing, and no assistance, startup
randomization, observation noise or pushes. The common fall rule is anchor
height error above 0.25 m or projected-gravity z error above 0.8. Trials stop
before reference wrapping, and all failures remain in completion denominators.
These thresholds define the protocol; they are not claims about deployment safety.

The single-world evaluation uses explicit contact capacity 128, constraint
capacity 1024, and contact-sensor match capacity 128. It rejects broadphase,
contact and constraint overflow rather than accepting a truncated collision
solve. The cluster monitor also makes every reported overflow fatal. Earlier
protocol results with overflow messages must be treated as invalid and rerun.

``protocol.json``, ``trials.jsonl`` and ``results.json`` record the schedule,
hashes, per-trial status, coverage, tracking errors, and physical telemetry.
Motor work and passive dissipation are integrated over physics substeps;
foot slip/contact force and joint-limit violations are sampled at 50 Hz.
Per-trial state/action traces support matched short-trajectory comparisons.
``initial_states.npz`` records physical starts. Failed trajectories contribute
their observed errors, and missing coverage is reported explicitly.

The evaluator exports into its own directory and compares ONNX actions against
PyTorch on ordinary and boundary observations (absolute tolerance 1e-5,
relative tolerance 1e-4). ``--backend onnx`` also drives the simulation with
the exported policy. ``export_parity.npz`` retains tested observations/actions.
The runtime is ONNX's reference evaluator with an ELU implementation that
evaluates the exponential only for nonpositive inputs. This preserves ELU
semantics and avoids overflow in its unused positive branch; simulator overflow
and nonfinite actions still fail evaluation. Results record this implementation.
Training-set
tracking and numerical survival are separate results; this protocol does not
establish held-out generalization or deployment-realistic state estimation.

Pass ``--onnx-file /path/to/params/policy.onnx`` to validate an existing training
export against the same checkpoint as well as the freshly exported policy.
``run_checked.py --evaluate`` supplies the training export automatically.
Historical checkpoints can additionally be evaluated with
``--actuator-model historical_unbounded`` to reproduce the original bypassed
torque envelope. This mode is evaluation-only; label its results separately from
the default corrected actuator and from newly trained corrected policies.

The evaluator saves the exact nominal model as ``model.mjb.gz`` with a checksum.
Render saved full-clip trajectories without running the policy or physics again:

.. code-block:: bash

   MUJOCO_GL=egl uv run python -m wbc_mjlab.scripts.render_evaluation \
     --evaluation /path/to/evaluation --output /path/to/videos

Rendering requires a working EGL graphics context. Videos use the same camera
settings and 50 Hz playback and display the task, clip, actuator model and trial
outcome. ``videos.json`` records outputs; ``--clip`` selects an exact clip name.

October 2026 campaign diagnosis
-------------------------------

The October 8 seed-42 campaign completed both EE tasks; G1, G1-SE, Zest, and
Zest-SE exited on nonfinite rollout observations. Their latest saved checkpoints
and the motion arrays were finite. G1 and G1-SE had unusually large assistive
torque normalization statistics, but the original runs did not save physics
NaN dumps. Frozen-policy replay did not reproduce a natural crash, so the
initial trigger of all four failures remains unproven.

Controlled tests reproduced two reset defects: cached wrench observations
survived resets, and clip wraps exposed pre-reset body kinematics. Both are now
covered by simulator regression tests and corrected. The cluster diagnostics
above are required for a subsequent campaign to establish whether any other
numerical failure remains. Short validation runs cannot establish stability for
200,000 iterations.

The revised October 14 campaign exposed a remaining G1-SE physics failure near
iteration 186125 despite those reset fixes. A timestamped `investigation handoff
and evidence bundle <https://github.com/RX-00/wbc-mjlab/tree/main/docs/investigations/2026-10-16-g1-se-nonfinite>`_
contains the state dump, matching compressed MuJoCo model, failure logs,
configuration, artifact hashes, and investigation limits. G1 completed the revised
200,000-iteration run. Other task statuses in that bundle are historical snapshots.
