#!/usr/bin/env python3
"""The wrist-image student: a small CNN on the wrist frame plus the proprioceptive
vector, regressing the teacher's action. Train on collected shards; DAgger
rounds re-collect with the student driving and the teacher labelling.

    sim/.venv-lerobot/bin/python sim/grasp/student.py train --data sim/datasets/grasp_student/r0 \\
        --out sim/runs/grasp_student/r0 --epochs 30
    sim/.venv-lerobot/bin/python sim/grasp/student.py train --data sim/datasets/grasp_student/r0 \\
        sim/datasets/grasp_student/r1 --init sim/runs/grasp_student/r0/student.pt --out sim/runs/grasp_student/r1
    sim/.venv-lerobot/bin/python sim/grasp/student.py eval sim/runs/grasp_student/r1/student.pt --n 30

Loss: Huber on the four motion actions plus the gripper target, all in
[-1, 1]. Augmentation: random crop-and-resize, brightness / contrast
jitter, so the real wrist camera's exposure and small pose errors do not
break it.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


class StudentNet(nn.Module):
    def __init__(self, obs_dim, act_dim=5, width=32):
        super().__init__()
        w = width
        self.cnn = nn.Sequential(
            nn.Conv2d(3, w, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(w, 2 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(2 * w, 4 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(4 * w, 4 * w, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten())
        feat = 4 * w * 12
        self.prop = nn.Sequential(nn.Linear(obs_dim, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(feat + 128, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                  nn.Linear(256, act_dim))

    def forward(self, img, obs):
        x = self.cnn(img)
        return torch.tanh(self.head(torch.cat([x, self.prop(obs)], -1)))


class ResNetStudent(nn.Module):
    """ImageNet-pretrained ResNet18 trunk (what the pour ACT policies use) plus the proprio MLP."""

    def __init__(self, obs_dim, act_dim=5, pretrained=True):
        super().__init__()
        import torchvision
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=w)
        self.trunk = nn.Sequential(*list(r.children())[:-1], nn.Flatten())      # 512
        self.prop = nn.Sequential(nn.Linear(obs_dim, 128), nn.GELU())
        self.head = nn.Sequential(nn.Linear(512 + 128, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                  nn.Linear(256, act_dim))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img, obs):
        x = self.trunk((img - self.mean) / self.std)
        return torch.tanh(self.head(torch.cat([x, self.prop(obs)], -1)))


class Student:
    def __init__(self, obs_dim, act_dim=5, width=32, device="cpu", arch="cnn"):
        self.arch = arch
        self.net = (ResNetStudent(obs_dim, act_dim) if arch == "resnet18"
                    else StudentNet(obs_dim, act_dim, width)).to(device)
        self.device = device
        self.obs_dim, self.act_dim, self.width = obs_dim, act_dim, width
        self.obs_mean = np.zeros(obs_dim, np.float32); self.obs_std = np.ones(obs_dim, np.float32)

    def _prep(self, img, obs):
        x = torch.as_tensor(np.asarray(img), device=self.device).permute(0, 3, 1, 2).float() / 255.0
        o = torch.as_tensor((np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std, device=self.device)
        return x, o

    @torch.no_grad()
    def act(self, img, obs):
        self.net.eval()
        x, o = self._prep(img, obs)
        return self.net(x, o).cpu().numpy()

    def save(self, path):
        torch.save(dict(net=self.net.state_dict(), obs_dim=self.obs_dim, act_dim=self.act_dim, width=self.width,
                        arch=self.arch, obs_mean=self.obs_mean, obs_std=self.obs_std), path)

    @classmethod
    def load(cls, path, device="cpu"):
        s = torch.load(path, map_location=device, weights_only=False)
        st = cls(s["obs_dim"], s["act_dim"], s["width"], device, s.get("arch", "cnn"))
        st.net.load_state_dict(s["net"]); st.obs_mean, st.obs_std = s["obs_mean"], s["obs_std"]
        return st


def load_shards(dirs):
    imgs, obs, acts = [], [], []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "ep*.npz"))):
            z = np.load(f)
            imgs.append(z["img"]); obs.append(z["obs"]); acts.append(z["act"])
    return np.concatenate(imgs), np.concatenate(obs), np.concatenate(acts)


def augment(x, rng):
    """x: (B, 3, H, W) float in [0, 1]. Crop-and-resize, brightness, contrast."""
    B, _, H, W = x.shape
    out = torch.empty_like(x)
    for i in range(B):
        s = rng.uniform(0.85, 1.0)
        h, w = int(H * s), int(W * s)
        y0, x0 = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
        crop = x[i:i + 1, :, y0:y0 + h, x0:x0 + w]
        out[i] = F.interpolate(crop, size=(H, W), mode="bilinear", align_corners=False)[0]
    b = torch.as_tensor(rng.uniform(-0.15, 0.15, (B, 1, 1, 1)), dtype=x.dtype, device=x.device)
    c = torch.as_tensor(rng.uniform(0.7, 1.3, (B, 1, 1, 1)), dtype=x.dtype, device=x.device)
    return ((out - 0.5) * c + 0.5 + b).clamp(0, 1)


def train(a):
    dev = a.device
    imgs, obs, acts = load_shards(a.data)
    print(f"{len(imgs)} ticks from {len(a.data)} shard dir(s); image {imgs.shape[1:]}")
    st = Student.load(a.init, dev) if a.init else Student(obs.shape[1], acts.shape[1], a.width, dev, a.arch)
    if not a.init:
        st.obs_mean = obs.mean(0).astype(np.float32); st.obs_std = (obs.std(0) + 1e-3).astype(np.float32)
    n = len(imgs); idx = np.arange(n); rng = np.random.default_rng(0); rng.shuffle(idx)
    n_val = max(256, n // 20); val, tr = idx[:n_val], idx[n_val:]
    opt = torch.optim.AdamW(st.net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    os.makedirs(a.out, exist_ok=True)
    x_val, o_val = st._prep(imgs[val], obs[val]); y_val = torch.as_tensor(acts[val], device=dev)
    best = float("inf")
    for ep in range(a.epochs):
        st.net.train(); rng.shuffle(tr); t0 = time.time(); tot = 0.0; nb = 0
        for i in range(0, len(tr), a.batch):
            b = tr[i:i + a.batch]
            x, o = st._prep(imgs[b], obs[b]); y = torch.as_tensor(acts[b], device=dev)
            x = augment(x, rng)
            loss = F.smooth_l1_loss(st.net(x, o), y, beta=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        sched.step()
        st.net.eval()
        with torch.no_grad():
            pv = st.net(x_val, o_val)
            vl = float(F.smooth_l1_loss(pv, y_val, beta=0.1))
            grip_acc = float(((pv[:, 4] < -0.2) == (y_val[:, 4] < -0.2)).float().mean())
        if vl < best:
            best = vl; st.save(os.path.join(a.out, "student.pt"))
        print(f"epoch {ep + 1:3d}: train {tot / nb:.4f}  val {vl:.4f}  grip-agree {grip_acc:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    print("best val", best, "->", os.path.join(a.out, "student.pt"))


def evaluate(a):
    from grasp.env import GraspEnv
    from collect import WRIST_SPEC
    st = Student.load(a.policy)
    env = GraspEnv(seed=a.seed, pool=a.n, wrist=a.wrist or WRIST_SPEC, wrist_size=tuple(a.size),
                   composites=a.composites, disturb_p=a.disturb, force_p=a.force,
                   wrist_jitter=tuple(a.jitter))
    ok, att = 0, []
    for ep in range(a.n):
        obs, info = env.reset(seed=a.seed + ep); done = False
        while not done:
            act = st.act(env.render_wrist()[None], env.student_obs()[None])[0]
            obs, r, term, trunc, info = env.step(act); done = term or trunc
        ok += info["success"]; att.append(info["attempts"])
        print(f"seed {a.seed + ep}: {info['outcome']:8s} attempts {info['attempts']} ticks {env.steps:3d} {info['kind']}")
    print(f"student success {ok}/{a.n}, mean attempts {np.mean(att):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--init")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--width", type=int, default=32)
    t.add_argument("--arch", default="cnn", choices=["cnn", "resnet18"])
    t.add_argument("--device", default="cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
    e = sub.add_parser("eval")
    e.add_argument("policy")
    e.add_argument("--n", type=int, default=30)
    e.add_argument("--seed", type=int, default=50_000)
    e.add_argument("--size", type=int, nargs=2, default=(128, 96))
    e.add_argument("--wrist")
    e.add_argument("--composites", type=float, default=0.5)
    e.add_argument("--disturb", type=float, default=0.3)
    e.add_argument("--force", type=float, default=0.3)
    e.add_argument("--jitter", type=float, nargs=3, default=(3.0, 0.7, 0.3), metavar=("MM", "DEG", "FOVY"),
                   help="per-scene wrist camera jitter; 0 0 0 pins it at the measured pose")
    a = ap.parse_args()
    (train if a.cmd == "train" else evaluate)(a)


if __name__ == "__main__":
    main()
