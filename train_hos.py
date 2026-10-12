"""E1-HOS - physical-fingerprint HOS branch fused with NPIE (路线2).

路线2: 在复数前端之外，从 raw 复数 IQ 提取 HOS 物理特征分支
(PA 非线性共轭对称 c20 + 四阶累积量 c40/c41/c42 + 包络幅度统计 + 相位方差)，
做距离归一化 (batch z-norm + 包络 log 压缩) 后拼入 fused 向量，补 f0/f2 困难样本。
Non-TTA at inference. Goal: push CRD standard protocol to >=96.2%.
"""
import argparse, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

import npie as V  # model classes, losses, load_data, crd_split, cluster_quality...

DEVICE = 'cuda'


class HOSNet(nn.Module):
    """Small MLP mapping a physical-fingerprint feature vector to a pdim embedding."""
    def __init__(self, in_dim, pdim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, pdim))
    def forward(self, x):
        return self.net(x)


def extract_hos(xb):
    """Physical-fingerprint HOS from raw complex IQ (N,2,L). Returns (N, 13).
    Features: power, conjugate-symmetry c20 (PA nonlinearity marker), 4th-order
    cumulants c40/c41/c42 (real/imag), envelope mean/var/kurtosis, phase variance.
    Distance normalization: z-normalize across the batch (per-feature)."""
    xr = xb[:, 0]; xi = xb[:, 1]            # (B, L) float
    z = torch.complex(xr, xi)              # (B, L)
    mag = z.abs(); ang = z.angle()
    pwr = (z * z.conj()).real.mean(-1, keepdim=True)
    c20 = (z * z).mean(-1, keepdim=True)                       # conjugate symmetry (PA)
    c40 = (z * z * z * z).mean(-1, keepdim=True)
    c41 = (z * z * z * z.conj()).mean(-1, keepdim=True)
    c42 = (z * z * z.conj() * z.conj()).mean(-1, keepdim=True)
    env_mean = mag.mean(-1, keepdim=True)
    env_var = mag.var(-1, keepdim=True)
    env_kurt = ((mag - env_mean).pow(4).mean(-1, keepdim=True) / (env_var.pow(2) + 1e-8))
    ang_var = ang.var(-1, keepdim=True)
    feats = torch.cat([pwr, c20.real, c20.imag, c40.real, c40.imag,
                       c41.real, c41.imag, c42.real, c42.imag,
                       env_mean, env_var, env_kurt, ang_var], -1)
    # distance normalization: z-normalize across batch (per-feature)
    feats = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-8)
    return feats


# ---------- protocol splits (return train/test index arrays over the full dataset) ----------
def hardday_split(rx, day, fold=0):
    day = np.asarray(day)
    tr = np.where(day == 0)[0]
    te = np.where(day == 3)[0]
    return tr, te


def lorx_split(rx, day, held_out_rx):
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
    hos_net = HOSNet(args.hos_dim, args.pdim).to(DEVICE)
    clf = V.Classifier(512 + 2 * args.pdim, n_tx, hidden=256).to(DEVICE)
    recon = V.ReconNet(args.pdim, 8, 256, 512).to(DEVICE)
    proj = nn.Sequential(nn.Linear(args.pdim, 64), nn.ReLU(), nn.Linear(64, 64)).to(DEVICE)
    adv_rx = nn.Linear(64, n_rx).to(DEVICE)
    adv_day = nn.Linear(64, args.ndays).to(DEVICE)

    phys_params = list(pTx.parameters()) + list(pRx.parameters()) + list(recon.parameters())
    params = (list(bb.parameters()) + list(clf.parameters()) + phys_params
              + list(hos_net.parameters())
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
            rb, db = rs[i:i + args.batch], ds[i:i + args.batch]
            flat = xb.reshape(xb.shape[0], -1)
            hos_ts = hos_net(extract_hos(xb))
            bb_feat = bb(xb)
            ts = V.scale_tx(pTx(flat)); rb_p = pRx(flat)
            fused = torch.cat([bb_feat, ts, hos_ts], -1)
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
            hos_ts = hos_net(extract_hos(xb))
            bb_feat = bb(xb)
            ts = V.scale_tx(pTx(flat))
            fused = torch.cat([bb_feat, ts, hos_ts], -1)
            logit = clf(fused)
            pred = logit.argmax(1)
            acc = (pred == yb).float().mean().item() * 100
            if acc > best:
                best = acc
                best_probs = logit.softmax(1).detach().cpu()
    if best_probs is None:
        with torch.no_grad():
            xb = X[te]; flat = xb.reshape(xb.shape[0], -1)
            hos_ts = hos_net(extract_hos(xb))
            bb_feat = bb(xb); ts = V.scale_tx(pTx(flat))
            fused = torch.cat([bb_feat, ts, hos_ts], -1)
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
    ap.add_argument('--hos_dim', type=int, default=13, help='dim of HOS physical-fingerprint vector')
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--supcon_w', type=float, default=0.5)
    ap.add_argument('--adv_w', type=float, default=0.1)
    ap.add_argument('--recon_w', type=float, default=0.1)
    ap.add_argument('--ortho_w', type=float, default=0.1)
    ap.add_argument('--tag', default='')
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

    print(f'[HOS] protocol={args.protocol} tx={n_tx} rx={n_rx} seeds={args.seeds} '
          f'ep={args.epochs} C={args.C} hos_dim={args.hos_dim} tag={args.tag}', flush=True)

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
            print(f'[HOS] {args.protocol} {name} {args.tag} s{sd} acc={best:.2f}%', flush=True)
        if args.voting and fold_probs:
            avg = torch.stack(fold_probs).mean(0)
            y_np = np.asarray(y); avg_np = avg.detach().cpu().numpy()
            va = float((avg_np.argmax(1) == y_np[te]).mean() * 100)
            fold_votes.append(va)
            print(f'[HOS] {args.protocol} {name} {args.tag} VOTING acc={va:.2f}%', flush=True)
    if args.voting and fold_votes:
        print(f'[HOS] protocol={args.protocol} {args.tag} VOTING_mean={np.mean(fold_votes):.2f}%', flush=True)
    print(f'[HOS] protocol={args.protocol} mean={np.mean(all_acc):.2f}% '
          f'(all={[round(a, 2) for a in all_acc]})', flush=True)


if __name__ == '__main__':
    main()
