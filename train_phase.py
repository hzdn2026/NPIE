"""E1-PHASE - phase-jitter augmentation variant of train_base.py (NPIE core in npie.py).

路线B: 训练时对复数 IQ 输入乘随机相位旋转 e^{j*phi} (phi~U(0,2*pi))，逼旋转不变特征，
直接对抗跨天相位漂移（弱折 f0/f2 的主因）。测试时不旋转（非 TTA）。
Goal: push CRD standard protocol to >=96.2%.
"""
import argparse, os
import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import npie as V  # model classes, losses, load_data, crd_split, cluster_quality...

DEVICE = 'cuda'


def rotate_phase(xb):
    """Random phase rotation e^{j*phi} on complex (N,2,*) input. Training aug ONLY
    (test split is never rotated -> model must rely on learned rotation invariance)."""
    phi = torch.rand(xb.size(0), *([1] * (xb.dim() - 1)), device=xb.device) * (2 * math.pi)
    c = torch.cos(phi); s = torch.sin(phi)
    xr = xb[:, 0:1]; xi = xb[:, 1:2]
    return torch.cat([xr * c - xi * s, xr * s + xi * c], dim=1)


# ---------- protocol splits (return train/test index arrays over the full dataset) ----------
def hardday_split(rx, day, fold=0):
    """Train on day0 only, test on day3 only. Ignores Rx fold (uses all Rx)."""
    day = np.asarray(day)
    tr = np.where(day == 0)[0]
    te = np.where(day == 3)[0]
    return tr, te


def lorx_split(rx, day, held_out_rx):
    """Leave-one-Rx-out: train on all samples except held_out_rx, test on held_out_rx."""
    rx = np.asarray(rx)
    tr = np.where(rx != held_out_rx)[0]
    te = np.where(rx == held_out_rx)[0]
    return tr, te


# ---------- train_one (mirrors v9f scale-up exactly, protocol-parameterized) ----------
def train_one(split_tr, split_te, seed, args, X, y, rx, day_dev, n_tx, n_rx):
    torch.manual_seed(seed)
    bb = V.BlackBoxStream(C=args.C, L=256, emb_dim=512).to(DEVICE)
    pTx = V.ParamNet(512, 256, args.pdim).to(DEVICE)
    pRx = V.ParamNet(512, 256, 8).to(DEVICE)
    clf = V.Classifier(512 + args.pdim, n_tx, hidden=256).to(DEVICE)
    recon = V.ReconNet(args.pdim, 8, 256, 512).to(DEVICE)
    proj = nn.Sequential(nn.Linear(args.pdim, 64), nn.ReLU(), nn.Linear(64, 64)).to(DEVICE)
    adv_rx = nn.Linear(64, n_rx).to(DEVICE)
    adv_day = nn.Linear(64, args.ndays).to(DEVICE)

    phys_params = list(pTx.parameters()) + list(pRx.parameters()) + list(recon.parameters())
    params = (list(bb.parameters()) + list(clf.parameters()) + phys_params
              + list(proj.parameters()) + list(adv_rx.parameters()) + list(adv_day.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    tr = torch.from_numpy(split_tr).long(); te = torch.from_numpy(split_te).long()
    Xtr = X[tr]; ytr = y[tr].to(DEVICE)
    rxtr = torch.from_numpy(rx[tr.numpy()]).to(DEVICE)
    daytr = day_dev[tr]
    best = 0
    best_probs = None
    for ep in range(args.epochs):
        warmup = ep < args.warmup
        lambd = ep / max(args.epochs - 1, 1)
        for p in phys_params:
            p.requires_grad_(not warmup)
        perm = torch.randperm(Xtr.shape[0])
        Xs, ys, rs, ds = Xtr[perm], ytr[perm], rxtr[perm], daytr[perm]
        for i in range(0, Xtr.shape[0], args.batch):
            xb, yb = Xs[i:i + args.batch], ys[i:i + args.batch]
            if getattr(args, 'phase_aug', False):
                xb = rotate_phase(xb)
            rb, db = rs[i:i + args.batch], ds[i:i + args.batch]
            flat = xb.reshape(xb.shape[0], -1)
            bb_feat = bb(xb)
            ts = V.scale_tx(pTx(flat)); rb_p = pRx(flat)
            fused = torch.cat([bb_feat, ts], -1)
            L_main = crit(clf(fused), yb)
            if warmup:
                loss = L_main
                L_supcon = L_adv = L_recon = L_ortho = torch.zeros(1, device=DEVICE)
            else:
                zp = proj(ts)
                L_supcon = V.supcon_loss(zp, yb)
                rev = V.GRL.apply(zp, lambd)
                L_adv = crit(adv_rx(rev), rb) + crit(adv_day(rev), db)
                L_recon = F.mse_loss(recon(ts, rb_p), flat) / (flat.var() + 1e-8)
                L_ortho = V.ortho_loss(ts, rb_p)
                loss = (L_main + args.supcon_w * L_supcon + args.adv_w * L_adv
                        + args.recon_w * L_recon + args.ortho_w * L_ortho)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        with torch.no_grad():
            xb = X[te]; yb = y[te].to(DEVICE)
            flat = xb.reshape(xb.shape[0], -1)
            bb_feat = bb(xb)
            ts = V.scale_tx(pTx(flat))
            fused = torch.cat([bb_feat, ts], -1)
            logit = clf(fused)
            pred = logit.argmax(1)
            acc = (pred == yb).float().mean().item() * 100
            if acc > best:
                best = acc
                best_probs = logit.softmax(1).detach().cpu()
    if best_probs is None:
        with torch.no_grad():
            xb = X[te]; flat = xb.reshape(xb.shape[0], -1)
            bb_feat = bb(xb); ts = V.scale_tx(pTx(flat))
            fused = torch.cat([bb_feat, ts], -1)
            best_probs = clf(fused).softmax(1).detach().cpu()
    return best, best_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data_cache/wisig_manySig_raw_full.npz')
    ap.add_argument('--protocol', default='crd', choices=['crd', 'hardday', 'lorx'])
    ap.add_argument('--folds', default='0,1,2,3')
    ap.add_argument('--lorx_heldout', default='0,4,8', help='comma Rx ids for leave-one-Rx-out')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--C', type=int, default=32)
    ap.add_argument('--pdim', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--supcon_w', type=float, default=0.5)
    ap.add_argument('--adv_w', type=float, default=0.1)
    ap.add_argument('--recon_w', type=float, default=0.1)
    ap.add_argument('--ortho_w', type=float, default=0.1)
    ap.add_argument('--tag', default='')
    ap.add_argument('--phase_aug', action='store_true',
                    help='random phase rotation augmentation on complex input (train only)')
    ap.add_argument('--voting', action='store_true', help='in-process soft-voting over seeds of this config')
    ap.add_argument('--save_probs', action='store_true', help='save per-(fold,seed,tag) test softmax to probs/*.npy')
    args = ap.parse_args()

    iq, tx, rx, day = V.load_data(args.npz)
    X = torch.from_numpy(iq).to(DEVICE)
    y = torch.from_numpy(tx)
    day_dev = torch.from_numpy(day).to(DEVICE)
    n_tx = int(tx.max()) + 1
    n_rx = int(rx.max()) + 1
    rx_np = rx if isinstance(rx, np.ndarray) else rx.numpy()
    day_np = day if isinstance(day, np.ndarray) else day.numpy()
    n_days = int(day_np.max()) + 1
    args.ndays = n_days

    print(f'[PHASE] protocol={args.protocol} tx={n_tx} rx={n_rx} seeds={args.seeds} '
          f'ep={args.epochs} C={args.C} phase_aug={args.phase_aug} tag={args.tag}', flush=True)

    splits = []
    if args.protocol == 'crd':
        for f in [int(x) for x in args.folds.split(',')]:
            tr, te = V.crd_split(rx_np, day_np, f)
            splits.append((f'f{f}', tr, te))
    elif args.protocol == 'hardday':
        tr, te = hardday_split(rx_np, day_np)
        splits.append(('hardday', tr, te))
    elif args.protocol == 'lorx':
        for r in [int(x) for x in args.lorx_heldout.split(',')]:
            tr, te = lorx_split(rx_np, day_np, r)
            splits.append((f'lorx-r{r}', tr, te))

    seeds = [int(s) for s in args.seeds.split(',')]
    all_acc = []
    fold_votes = []
    for name, tr, te in splits:
        fold_probs = []
        fold_acc = []
        for sd in seeds:
            best, probs = train_one(tr, te, sd, args, X, y, rx_np, day_dev, n_tx, n_rx)
            fold_acc.append(best); all_acc.append(best)
            if args.save_probs:
                os.makedirs('probs', exist_ok=True)
                np.save(f'probs/{args.protocol}_{name}_{args.tag}_s{sd}.npy', probs.numpy())
            if args.voting:
                fold_probs.append(probs)
            print(f'[PHASE] {args.protocol} {name} {args.tag} s{sd} acc={best:.2f}%', flush=True)
        if args.voting and fold_probs:
            avg = torch.stack(fold_probs).mean(0)
            y_np = np.asarray(y); avg_np = avg.detach().cpu().numpy()
            va = float((avg_np.argmax(1) == y_np[te]).mean() * 100)
            fold_votes.append(va)
            print(f'[PHASE] {args.protocol} {name} {args.tag} VOTING acc={va:.2f}%', flush=True)
    if args.voting and fold_votes:
        print(f'[PHASE] protocol={args.protocol} {args.tag} VOTING_mean={np.mean(fold_votes):.2f}%', flush=True)
    print(f'[PHASE] protocol={args.protocol} mean={np.mean(all_acc):.2f}% '
          f'(all={[round(a, 2) for a in all_acc]})', flush=True)


if __name__ == '__main__':
    main()
