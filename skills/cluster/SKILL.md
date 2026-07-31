---
name: cluster
description: Run GPU workloads on the Construct Labs cluster via SkyPilot. Use when launching training or inference for the hackathon, debugging a pending or evicted job, or checking what is running.
---

# Cluster access

SkyPilot is the only entrypoint. No kubeconfig, no kubectl, no node SSH. One endpoint, one
identity per person or agent, everything visible in one dashboard.

> Event-specific values your organizers give you:
> `SKYPILOT_API_SERVER_ENDPOINT`, your team's queue name, and the training image digest.
> Fill them in below before running anything.

## Two hard rules

The GPUs are shared across every team for the whole event. Both of these are about not
taking capacity away from the room.

**1. Submit jobs. Do not create clusters.**

Use `sky jobs launch`. Do not use `sky launch -c <name>`.

`sky launch -c <name>` creates a **persistent cluster** that holds its GPUs from the moment
it starts until someone runs `sky down`. It keeps holding them while you read logs, while
you think, while you are at lunch, and overnight. A managed job with `sky jobs launch`
takes GPUs when it runs and gives them back when it finishes.

One idle dev cluster can park 8 H100s for a day. There are not enough GPUs for that to
happen more than once.

**2. Single node only. No multi-node runs.**

Every task file stays at one node. Do not set `num_nodes`, do not launch anything that
spans hosts.

Multi-node jobs are admitted all-or-nothing, so they sit holding nothing until an entire
second node frees up, and then take both at once. During an event with many small jobs
that means one team blocks the queue for everyone while getting nothing done itself. The
training config that ships with this repo is single node, 8x H100, on purpose.

If you genuinely believe you need more than one node, talk to an organizer first.

## Setup, once

1. Install Tailscale and join the tailnet with the account the organizers tell you to use.
   Verify: `tailscale status` lists the cluster head, and the dashboard URL loads in a
   browser. A login prompt means success. If neither works you are on the wrong tailnet,
   which is the single most common setup failure.
2. `pip install "skypilot[kubernetes]"`
3. `export SKYPILOT_API_SERVER_ENDPOINT=https://<user>:<pass>@<head>.ts.net`
4. Verify: `sky api info` shows the server, `sky gpus list` shows H100s.

Agents use a service-account token in the same variable, owned by a human.

## Recipes

**Everything is a managed job.** One shape, single node:

```yaml
# task.yaml
name: myjob
resources:
  infra: kubernetes
  accelerators: H100-80GB:1     # ask for what you need, not the whole node
run: |
  python my_script.py
```

```bash
sky jobs launch -y -n myjob task.yaml
sky jobs queue                   # your jobs and their state
sky jobs logs myjob              # or --no-follow for a snapshot
sky jobs cancel myjob
```

**Iterating on code.** The instinct is to grab an interactive box. Do not. Put your
debugging in the `run:` block and resubmit: a job that starts, prints, and exits takes
seconds of GPU time instead of parking a node while you edit. `workdir:` syncs your local
directory each launch, so the edit-resubmit loop is fast.

```yaml
workdir: .
run: |
  python -c "import torch; print(torch.cuda.device_count())"
```

**Short probes are cheap and encouraged.** Verifying an import, a checkpoint path, or a
config takes a one-CPU or one-GPU job of a few seconds. Do that before submitting anything
long, every time.

**RL training on this environment.** See [`../../training/README.md`](../../training/README.md).

## Non-negotiables

- **`priorityClassName: train` in `pod_config`.** Without it your pods sit at priority 0
  and get preempted by anything else in the queue, including a one-GPU probe. This has
  killed a running 16-GPU job.
- **No `sky launch -c`, no `num_nodes`.** See the two hard rules above. If you already
  created a cluster, `sky status` will show it and `sky down <name>` releases it. Do that
  now rather than at the end of the day.
- **`/mnt/nvme` is node-local.** A dataset or config staged on one node does not exist on
  another. Ship files with `file_mounts`, not by assuming a shared path.
- **Unique `output_dir` per job.** Two jobs writing one output dir kills one of them at
  its next checkpoint.

## Checking on things, non-interactively

```bash
sky api info                    # connectivity
sky gpus list                   # capacity
sky jobs queue                  # your jobs
sky jobs logs <name> --no-follow
sky status                      # clusters YOU are holding — should be empty
```

`sky status` should show nothing. If it lists a cluster, you are holding GPUs the rest of
the room cannot use: `sky down <name>`.

If your job is Pending, you are over your queue's quota and it will start when capacity
frees. Do not retry-loop on Pending.

## Failures and what they mean

| Symptom | Cause and fix |
|---|---|
| Dashboard or endpoint unreachable | You are on a personal tailnet. Switch to the event one. |
| `ErrImagePull ... 403` | The pull secret must exist in the `skypilot` namespace and your task must list it in `imagePullSecrets`. Ask an organizer. |
| Pod stuck pulling a cached image | Set `imagePullPolicy: IfNotPresent`. A `:latest` tag defaults to `Always` and can trickle for hours. |
| Job evicted mid-run, preemptee priority 0 | Missing `priorityClassName: train`. |
| Files missing inside the pod | `/mnt/nvme` is node-local. |
| Managed job FAILED but a pod still runs | `sky jobs cancel` the job. Deleting the pod directly gets it resurrected by the controller. |
| Multiple vLLM servers on one node crash | Give each its own `--server.port`, `--data-parallel-rpc-port`, and `VLLM_PORT`, and stagger boots. |

## Long runs

Everything here is a managed job already, but a run long enough to leave alone needs
recovery configured, or it will not survive a preemption:

```yaml
resources:
  job_recovery:
    max_restarts_on_errors: 3
```

plus `--ckpt.resume-step -1` in the run command. An evicted pod looks like a program
failure to the controller, and a restarted attempt without resume refuses the checkpointed
output dir and burns a restart. Managed attempts reuse the same pod name, so distinguish
attempts by creation time.
