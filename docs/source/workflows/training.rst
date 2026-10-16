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

``scripts/cluster/`` provides an Apptainer build and independent Slurm jobs for
the six WBC/Zest-family G1 tasks. The launcher defaults to the A2R cluster's
``debug`` partition, one RTX 5090, eight CPUs, 16 GiB RAM, and exclusion of
``a2r-main``. Adapt these resource directives for another cluster.

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

These are six separate jobs; a failed task does not cancel another. Every full
run starts fresh with seed 42, 2,048 environments, 24 rollout steps, 200,000 PPO
iterations, and checkpoint interval 5,000. The sample motions are loaded from the
snapshot. All tasks use ``njmax=1024`` to provide more constraint capacity than
the G1 default of 250, and ``contact_sensor_maxmatch=128`` (the original sample
campaign reported a required capacity of 65). The runner rejects reported
``nefc``, ``ncon``, and contact-match overflows,
propagates training failures, and verifies the final checkpoint, saved common
settings, and exported artifacts. Smoke results are separate and are never
loaded into full training.

The optional ``stability`` phase runs 1,000 fresh PPO iterations per task. Run
smoke and stability checks to completion before submitting full training; the
launcher does not automatically gate one phase on another.

Cluster jobs enable per-term observation checks with ``nan_policy=error`` for
both actor and critic, and a 32-state physics NaN guard. A physics NaN/Inf dump
also fails the job, even if an environment reset would otherwise hide it.
Dumps and model files persist under
``diagnostics/<phase>_<task>_job<id>/`` inside the campaign. The JSON result records
this directory. These checks stop on invalid values; they do not sanitize them.

``jobs_smoke.tsv``, ``jobs_stability.tsv``, and ``jobs_train.tsv`` record task/job IDs. Slurm output is in
``console/``, training outputs are in ``smoke/``, ``stability/``, or ``train/``, and finished jobs
write JSON results in ``status/``. Missing JSON may indicate that Slurm killed a
job before the wrapper finished; always also check ``sacct``. A normal fresh
200,000-iteration run ends at zero-based checkpoint ``model_199999.pt``.

Tips
----

- **Fewer envs for debugging:** ``--env.scene.num-envs 512``
- **Extend horizon:** ``--agent.max-iterations 300000`` (works with resume)
- **Faster restarts on big libraries:** ``--cache-motion-bundle``

.. seealso::

   - :doc:`deploy` — export the trained policy for a robot
   - :doc:`../concepts/adaptive_sampling` — what the sampling curves show
   - :ref:`cli-train` — every option

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
