---
name: cluster
description: Run GPU workloads on the Construct Labs cluster via SkyPilot. Use when bringing up your team's GPU box, running training/inference on it, persisting data, SSHing in, or debugging access.
---

# Cluster access — your team's persistent GPU box

SkyPilot is the only entrypoint. No kubeconfig, no kubectl, no node SSH. Your team gets
**one persistent GPU box** in a private `hackathon` workspace: you bring it up once, work on
it all event, and everything you write on it stays there until you tear it down.

> Values your organizers give you (fill these in before running anything):
> the **Tailscale auth key**, your **team login** (`team-N` + password), and the **training
> image digest**. One login per team — everyone on the team uses the same one.

## 1. Get on our network + connect (once per person)

1. Install Tailscale and join with the key the organizers give you:
   ```bash
   curl -fsSL https://tailscale.com/install.sh | sudo sh
   sudo tailscale up --authkey='<the tag:hackathon key>'
   ```
   Verify: `tailscale status` lists `sky-head`, and `https://sky-head.taila766fb.ts.net/dashboard`
   loads in a browser (a login prompt = success). If neither works you are on the wrong tailnet —
   the single most common setup failure.
2. Install the client and point it at the server:
   ```bash
   pip install "skypilot[kubernetes]"
   export SKYPILOT_API_SERVER_ENDPOINT='https://team-N:<password>@sky-head.taila766fb.ts.net'
   ```
3. Verify: `sky api info` shows you as `team-N`. Your workspace (`hackathon`) is selected
   automatically — you never pass `--workspace`.

Everyone on your team runs the same three steps with the same credential. You all see and
share the same box.

## 2. Bring up your team box (once)

```bash
# From the training/ directory (run.yaml ships the env with the box):
sky launch -c team-N-box training/run.yaml
```

This is a **persistent cluster** (`-c <name>`). It holds its GPUs and keeps its disk from the
moment it starts until you run `sky down`. That is exactly what you want here: it is your box
for the event.

- **Default size is 4 GPUs** (what the training config uses). You *may* use more if the fleet is
  free — you share a **20-GPU pool** across all teams — but be considerate and leave room for
  others. There is no hard per-box cap; the pool is the limit, and a request that doesn't fit
  right now **queues** and starts when GPUs free (it does not fail).
- **One box per team.** Don't spin up extra clusters. `sky status` should show just your box.

## 3. Work on the box

```bash
sky exec team-N-box training/run.yaml    # run another job on the box (reuses it, no re-provision)
ssh team-N-box                           # interactive shell on the box
sky logs team-N-box                      # console output of the last job
sky queue team-N-box                     # jobs you've run on it
```

`sky exec` and `ssh` both land on the **same box** as your launch — same GPUs, same files.

## 4. Persist data on the box (this is the point of a persistent box)

Anything you write to the box's disk **stays there across jobs, `sky exec`, and SSH sessions**,
for as long as the box is up. Use the `/persist` volume the template mounts (or just your home
dir). Tested end-to-end: one job writes `/persist/…`, a later `sky exec` job reads and updates
it, and an SSH session sees the change.

```bash
# job A writes:            echo hi > /persist/checkpoint
# later `sky exec` job B:  cat /persist/checkpoint   # -> hi   (it's still there)
```

**Getting results off the box** (do this before you tear it down — the disk does NOT survive
`sky down`):
```bash
rsync -avP team-N-box:/persist/ ./results/     # SkyPilot tunnels it; no extra setup
```
For anything you must keep long-term, push it out to **your own** storage over the internet
(the box can reach the public net): Hugging Face Hub for model weights (`HF_TOKEN`), Weights &
Biases for metrics (`WANDB_API_KEY`) — set them in the task's `envs:`.

## 5. When you're done

```bash
sky down team-N-box     # frees your GPUs for other teams. Pull results off FIRST.
```

## The task shape (copy this — jobs fail without it)

```yaml
name: myjob
resources:
  infra: kubernetes
  accelerators: H100-80GB:4          # your box size (4 is the default; more only if free)
  # The prepared hackathon image (PUBLIC, anonymous pull — no credential). It already
  # contains prime-rl. Digest-pinned; if organizers announce a rotated digest, use that.
  image_id: docker:europe-west3-docker.pkg.dev/operator-agent-487820/ml-hackathon/prime-rl-base@sha256:a69f048650ac36d62da91effa337602c4541826558f7095eba9c47c433f7753b
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
              - {name: persist, mountPath: /persist}    # your durable-while-up workspace
        volumes:
          - {name: persist, emptyDir: {sizeLimit: 200Gi}}   # emptyDir, NOT hostPath
run: |
  echo "work here; write anything you want to keep to /persist"
```

## What's blocked, and why your box is still safe

Your box is a hardened container. From it (jobs **and** SSH alike) you **cannot**:
- mount our disks — `hostPath` / `/mnt/nvme` is forbidden (verified: they don't exist in the box);
- create a PersistentVolumeClaim;
- reach our internal network, other namespaces, or cloud metadata — **only the public internet**
  (PyPI / Hugging Face / W&B / GitHub) is reachable;
- run a different image (customize the approved one with `pip install` inside the box);
- touch the Kubernetes API (the box runs as a powerless, no-RBAC service account).

None of that limits normal work — install what you need, train, persist, pull results out.

## Failures and what they mean

| Symptom | Cause and fix |
|---|---|
| Dashboard/endpoint unreachable | You're on a personal tailnet. Switch to the event one. |
| Job dies in setup: `sudo: the "no new privileges" flag is set` | Missing `runAsUser: 0` in the container securityContext. |
| Pod rejected: `... hostPath volumes` | You used a `hostPath` (e.g. `/mnt/nvme`). Use the `/persist` `emptyDir`. |
| Pod rejected: `... may only run the approved image` | Wrong image. Use the digest-pinned `ml-hackathon/prime-rl-base`. |
| `sky launch` sits **Pending** | The 20-GPU pool is full right now; it starts when a box frees. Lower your GPU ask, or wait. |
| Files gone after `sky down` | The box disk doesn't survive teardown. Pull with `rsync` / push to HF/W&B **before** `sky down`. |
| `ErrImagePull` | You referenced a private image or added a pull secret. The hackathon image is public; drop `imagePullSecrets`. |

## RL training on this environment

See [`../../training/README.md`](../../training/README.md) — `training/run.yaml` is already in this
shape (approved image, `runAsUser: 0`, `/persist` volume) and brings the box up with training running.
