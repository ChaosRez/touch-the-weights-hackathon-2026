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

**Dev box, for debugging.** Holds its GPU until you release it, so always autostop.

```bash
sky launch -c dev --gpus H100-80GB:1 -i 30   # -i 30 = autostop after 30 idle minutes
ssh dev
sky down dev
```

**Batch job.**

```yaml
# task.yaml
name: myjob
resources:
  infra: kubernetes
  accelerators: H100-80GB:1
run: |
  python my_script.py
```

```bash
sky launch -c myjob -y --down task.yaml    # --down = tear down when finished
sky logs myjob
```

**Multi-node.** Set `num_nodes: N`. That is the whole recipe. Pods are admitted
all-or-nothing as one workload, and the API server attaches the queue labels itself. Do
not hand-write PodGroups, scheduler names, or queue labels. SkyPilot injects
`SKYPILOT_NODE_RANK`, `SKYPILOT_NODE_IPS`, `SKYPILOT_NUM_NODES` for torchrun.

**RL training on this environment.** See [`../../training/README.md`](../../training/README.md).

## Non-negotiables

- **`priorityClassName: train` in `pod_config`.** Without it your pods sit at priority 0
  and get preempted by anything else in the queue, including a one-GPU probe. This has
  killed a running 16-GPU job.
- **Always `--down` or `-i <minutes>`.** Never leave a cluster holding GPUs with no
  autostop. This matters more than usual with several teams sharing quota.
- **`/mnt/nvme` is node-local.** A dataset or config staged on one node does not exist on
  another. Ship files with `file_mounts`, not by assuming a shared path.
- **Unique `output_dir` per job.** Two jobs writing one output dir kills one of them at
  its next checkpoint.

## Checking on things, non-interactively

```bash
sky api info                 # connectivity
sky gpus list                # capacity
sky queue --all              # what is running
sky logs <cluster> --no-follow
sky jobs queue               # managed jobs
```

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

## Unattended runs

Submit with `sky jobs launch` and set both of these, or recovery will not work:

```yaml
resources:
  job_recovery:
    max_restarts_on_errors: 3
```

plus `--ckpt.resume-step -1` in the run command. An evicted pod looks like a program
failure to the controller, and a restarted attempt without resume refuses the checkpointed
output dir and burns a restart. Managed attempts reuse the same pod name, so distinguish
attempts by creation time.
