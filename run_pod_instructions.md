# Renting a GPU on RunPod (for the auth_projection experiments)

## Why RunPod (not Lambda)

Lambda Labs A100s are often sold out — capacity is genuinely constrained, sometimes on-and-off across the day. RunPod's Community Cloud has a much bigger supply pool (peer-hosted GPUs), so you almost always get a box on demand. RunPod is also ~33% cheaper for the same A100.

Trade-off: RunPod Community is community-hosted, so you occasionally get a flaky host (machine reboots, slow disk). For one-off 2-hour experiments, this is fine — if your pod misbehaves, terminate it and re-deploy.

## Step-by-step

1. **Sign up** at runpod.io, add a payment method, deposit at least $10 in credit.
2. **Add your SSH key** (one-time setup):
   - Account settings → SSH Public Keys → "Add SSH Key"
   - Paste the contents of your local `~/.ssh/id_rsa.pub` (or whichever public key you use)
   - Save
3. **Deploy a pod**:
   - Top-left → "Pods" → "Deploy"
   - GPU type: **A100 PCIe 40GB** (sufficient for Llama 3.1 8B inference; 80GB is overkill)
   - Cloud type: **Community Cloud** (cheaper)
   - Sort by price; pick the cheapest with a decent host rating
   - Template: **"RunPod PyTorch 2.4"** (or any PyTorch image — has python/ssh/git pre-installed)
   - Volume: 50GB is plenty (Llama 8B is ~16GB on disk + buffer for activations)
   - Click Deploy
4. **Wait ~2 minutes** for the pod to boot.
5. **Get the SSH command**:
   - In the pod's "Connect" panel
   - Look for **"SSH over exposed TCP"** (not the web terminal)
   - Copy the command — looks like:
     ```
     ssh root@<ip> -p <port> -i ~/.ssh/id_rsa
     ```
   - Paste this to Claude in the conversation; Claude takes over from there.

## Pricing snapshot (May 2026)

| GPU | RunPod Community | RunPod Secure |
|---|---|---|
| A100 40GB | ~$0.79/hr | ~$1.29/hr |
| A100 80GB | ~$1.19/hr | ~$1.89/hr |
| H100 80GB | ~$2.49/hr | ~$3.49/hr |

For the auth_projection 8B rerun: expect ~2 hours of compute = **~$1.60 on Community A100 40GB**.

## After Claude is done

Claude will:
- Run the experiments
- rsync results back to this local repo
- Run `sudo shutdown -h now` on the pod (to stop billing fast)

You should still:
- Manually terminate the pod in the RunPod web UI when you see it's done. The shutdown command stops it but RunPod still charges until you fully terminate.
- "Terminate Pod" → confirm → instance is destroyed and billing stops fully.

## Troubleshooting

- **"Permission denied (publickey)"**: SSH key didn't sync to the pod. Re-check that you added the key in account settings *before* deploying the pod (it has to be there at boot time). If you added it after, terminate the pod and re-deploy.
- **Pod stuck in "Initializing"**: rare, kill it and redeploy with a different host.
- **rsync slow**: Community Cloud hosts are scattered geographically; latency varies. Acceptable for our data size (~100MB total).
