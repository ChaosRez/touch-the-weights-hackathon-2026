---
name: cluster
description: Run GPU workloads on the Construct Labs cluster via SkyPilot. Use when launching training or inference for the hackathon, debugging a pending or evicted job, or checking what is running.
---

# Cluster access

SkyPilot is the only entrypoint. No kubeconfig, no kubectl, no node SSH. One endpoint, one
identity per person or agent, everything visible in one dashboard. You run in an isolated
`hackathon` workspace: you can see and manage only your own jobs, never anyone else's.

> Event-specific values your organizers give you:
> `SKYPILOT_API_SERVER_ENDPOINT` (with your `user:password`), the Tailscale login, and the
> **training image digest**. Fill them in before running anything.

## Setup, once

1. Install Tailscale and join the tailnet with the account/key the organizers give you.
   Verify: `tailscale status` lists the cluster head, and the dashboard URL loads in a
   browser (a login prompt = success). If neither works you are on the wrong tailnet — the
   single most common setup failure.
2. `pip install "skypilot[kubernetes]"` (or `uv pip install "skypilot[kubernetes]"`).
3. `export SKYPILOT_API_SERVER_ENDPOINT="https://<user>:<password>@<head>.ts.net"`
4. Verify: `sky api info` shows the server and your user; `sky gpus list` shows H100s.
   `sky status` should show **no clusters** — if it lists one, you are holding GPUs.

## Two hard rules

The GPUs are shared across every team. Both rules are about not taking capacity from the room.

**1. Submit jobs. Do not create clusters.** Use `sky jobs launch`, not `sky launch -c <name>`.
A named cluster holds its GPUs from the moment it starts until someone runs `sky down` —
while you read logs, think, or sleep. A managed job (`sky jobs launch`) gives them back when
it finishes.

**2. Single node only.** One node per task. Do not set `num_nodes`. Multi-node jobs are
admitted all-or-nothing and block the queue waiting for a whole second node.

## Your quota

- **4 GPUs, hard cap.** A job asking for more than `H100-80GB:4` is rejected by the namespace
  quota and sits **Pending forever** — it is not a queue wait, it will never start. Ask for
  what you need, up to 4.
- **Low priority, preemptible.** Your jobs run on spare capacity and are preempted when the
  hosts' owners reclaim it. For anything longer than a smoke, wire checkpoint/resume (see
  "Long runs"). You never preempt anyone else.

## The task shape (this matters — copy it)

Every guest task needs the same three things in `config.kubernetes.pod_config`, and jobs fail
without them. This is the minimal working shape:

```yaml
name: myjob
resources:
  infra: kubernetes
  accelerators: H100-80GB:1          # <= 4
  # The prepared hackathon image (PUBLIC, anonymous pull — no credential). Digest-pinned;
  # your organizers give you the current digest. It already contains prime-rl.
  image_id: docker:<registry>/ml-hackathon/prime-rl-base@sha256:<digest>
config:
  kubernetes:
    pod_config:
      spec:
        containers:
          - securityContext:
              runAsUser: 0             # REQUIRED. The image runs as a non-root user; SkyPilot
                                       # bootstraps ssh/ray as root. Omit this and your job dies
                                       # in setup with "sudo: no new privileges is set".
            volumeMounts:
              - {name: scratch, mountPath: /scratch}   # your writable workspace (caches, ckpts)
        volumes:
          - {name: scratch, emptyDir: {sizeLimit: 200Gi}}   # emptyDir, NOT hostPath
run: |
  python my_script.py
```

```bash
sky jobs launch -y -n myjob task.yaml
sky jobs queue                     # your jobs and their state
sky jobs logs myjob                # or --no-follow for a snapshot
sky jobs cancel myjob
```

## Non-negotiables (each of these is a job that fails without it)

- **`runAsUser: 0`** in the container `securityContext` — see above. The #1 thing people forget.
- **The public `ml-hackathon` image, pinned by digest.** You cannot pull our other images and
  you do not need a pull secret. Never `:latest` (an `IfNotPresent` launch can run a stale
  cached image for hours).
- **No `hostPath`.** The namespace forbids it (it is how you would reach the host's disks), so
  `/mnt/nvme` and host paths are unavailable. Use an `emptyDir` volume for scratch/caches, and
  point `HF_HOME`, `VLLM_CACHE_ROOT`, and your `output_dir` at it.
- **No `privileged`, no host namespaces, no extra images.** All rejected at admission; do not
  copy pod-security tricks from generic SkyPilot examples, they will not schedule here.
- **`sky jobs launch`, not `sky launch -c`. No `num_nodes`.** See the two hard rules.

## Checking on things, non-interactively

```bash
sky api info                    # connectivity + your identity
sky gpus list                   # capacity
sky jobs queue                  # your jobs (you see only your own)
sky jobs logs <name> --no-follow
sky status                      # clusters YOU are holding — should be empty
```

If your job is **Pending**: either you asked for >4 GPUs (it will never start — lower it), or
you are at your quota and it starts when capacity frees. Do not retry-loop on Pending.

## Failures and what they mean

| Symptom | Cause and fix |
|---|---|
| Dashboard or endpoint unreachable | You are on a personal tailnet. Switch to the event one. |
| Job dies in setup: `sudo: the "no new privileges" flag is set` | Missing `runAsUser: 0` in the container securityContext. |
| Pod rejected: `violates PodSecurity "baseline"... hostPath volumes` | You used a `hostPath` volume (e.g. `/mnt/nvme`). Use an `emptyDir` instead. |
| Pod rejected: `... may only run the approved hackathon image` | Wrong image. Use the digest-pinned `ml-hackathon/prime-rl-base` the organizers gave you. |
| Job stuck **Pending**, never schedules | You asked for more than 4 GPUs. The quota caps you at 4; lower `accelerators`. |
| `ErrImagePull` | You referenced a private image or added a pull secret. The hackathon image is public; drop `imagePullSecrets` and use the `ml-hackathon` digest. |
| Files missing inside the pod | You assumed a shared node path. Ship files with `file_mounts`; scratch is your `emptyDir`, not `/mnt/nvme`. |
| Managed job FAILED but a pod still runs | `sky jobs cancel` the job (deleting the pod directly gets it resurrected by the controller). |

## Long runs

Your jobs are preemptible, so anything long needs recovery configured or it won't survive:

```yaml
resources:
  job_recovery:
    max_restarts_on_errors: 3
```

plus `--ckpt.resume-step -1` in the run command. An evicted pod looks like a program failure
to the controller; a restart without resume refuses the checkpointed output dir and burns a
restart. Point `output_dir` at your `/scratch` emptyDir (it is per-pod and does not survive
teardown — copy anything you want to keep out with `sky` before the job ends, or push to your
own storage).

## RL training on this environment

See [`../../training/README.md`](../../training/README.md) — the `run.yaml` there is already in
this shape (public image, `runAsUser: 0`, emptyDir scratch, 4 GPUs).
