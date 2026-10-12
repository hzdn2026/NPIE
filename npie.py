"""v9f / NPIE-v3: FORCE theta_Tx to be a device fingerprint.

Diagnosis from v9e: theta_Tx does NOT cluster by device (intra/inter=8.53,
PCA shows device-colored points fully mixed) and the physical branch gives
no accuracy gain (-0.20pp). The params learned generic signal shape, not
device identity.

v9f adds three constraints on theta_Tx:
  1. SupCon loss  : same-device theta_Tx pulled together, different pushed
                    apart (device-level supervised contrastive).
  2. GRL adversary: theta_Tx -> predict Rx id / day id through a gradient
                    reversal layer, so theta_Tx becomes receiver/day INVARIANT
                    by construction (hard evidence for the paper).
  3. Centroid probe: after each run, classify test devices by nearest training
                    centroid in theta_Tx space -> direct "is it a fingerprint?"
                    number. Plus intra/inter cluster quality.

Black-box stream unchanged (v8/v9d winning complex front-end).
"""
import argparse, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

torch.backends.cudnn.benchmark = True


def load_data(npz):
    d = np.load(npz)
    iq = d['iq'].astype(np.float32)
    tx = d['tx'].astype(np.int64)
    rx = d['rx'].astype(np.int64)
    day = d['day'].astype(np.int64)
    return iq, tx, rx, day


def crd_split(rx, day, fold):
    urx = np.unique(rx)
    test_rx = set(urx[fold::4].tolist())
    tr = np.where((~np.isin(rx, list(test_rx))) & (day < 2))[0]
    te = np.where((np.isin(rx, list(test_rx))) & (day >= 2))[0]
    return tr, te


CFO_S = 0.5; IQI_S = 0.3; DC_S = 0.1; PHASE_S = math.pi


class ComplexConv1d(nn.Module):
    def __init__(self, C=8, K=31):
        super().__init__()
        self.wr = nn.Conv1d(1, C, K, padding=K // 2, bias=False)
        self.wi = nn.Conv1d(1, C, K, padding=K // 2, bias=False)
    def forward(self, x):
        xr = x[:, 0:1]; xi = x[:, 1:2]
        return torch.cat([self.wr(xr) - self.wi(xi), self.wr(xi) + self.wi(xr)], dim=1)


class BlackBoxStream(nn.Module):
    def __init__(self, C=8, L=256, emb_dim=512):
        super().__init__()
        self.fe = ComplexConv1d(C=C, K=31)
        self.proj = nn.Linear(2 * C * L, 512)
        self.bn1 = nn.BatchNorm1d(512); self.fc1 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512); self.fc2 = nn.Linear(512, emb_dim)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        z = self.fe(x)
        z = self.proj(z.reshape(z.size(0), -1))
        z = self.drop(F.relu(self.bn1(self.fc1(z))))
        return self.drop(F.relu(self.bn2(self.fc2(z))))


class ParamNet(nn.Module):
    def __init__(self, in_dim=512, hidden=256, pdim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, pdim))
    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, num, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden, num))
    def forward(self, x):
        return self.net(x)


class ReconNet(nn.Module):
    def __init__(self, ptx_dim=32, prx_dim=8, hidden=256, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ptx_dim + prx_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))
    def forward(self, ptx, prx):
        return self.net(torch.cat([ptx, prx], -1))


class GRL(torch.autograd.Function):
    """Gradient reversal: forward identity, backward negate."""
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def scale_tx(p):
    out = torch.zeros_like(p)
    out[:, 0] = torch.tanh(p[:, 0]) * CFO_S
    out[:, 1] = torch.tanh(p[:, 1]) * IQI_S
    out[:, 2] = torch.tanh(p[:, 2]) * DC_S
    out[:, 3] = torch.tanh(p[:, 3]) * DC_S
    out[:, 4] = torch.tanh(p[:, 4])
    out[:, 5] = torch.tanh(p[:, 5])
    out[:, 6] = torch.tanh(p[:, 6])
    out[:, 7] = torch.tanh(p[:, 7]) * PHASE_S
    out[:, 8:] = torch.tanh(p[:, 8:])
    return out


def ortho_loss(a, b):
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    c = torch.matmul(a.T, b) / a.shape[0]
    return (c ** 2).sum()


def supcon_loss(z, labels, temperature=0.1):
    """Supervised contrastive (Khosla et al.): same-label positives."""
    z = F.normalize(z, dim=1)
    sim = z @ z.T / temperature
    n = z.shape[0]
    diag = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (labels[:, None] == labels[None, :]) & ~diag
    sim_max = sim.masked_fill(diag, -1e9).max(1, keepdim=True)[0]
    logits = sim - sim_max
    logits = logits.masked_fill(diag, -1e9)
    logp = logits - torch.logsumexp(logits, 1, keepdim=True)
    cnt = pos.float().sum(1)
    loss = -(pos.float() * logp).sum(1) / cnt.clamp(min=1)
    return loss[cnt > 0].mean()


def cluster_quality(ts_np, y_np):
    """intra = mean same-device pairwise dist; inter = mean diff-device."""
    ts = F.normalize(torch.from_numpy(ts_np), dim=1).numpy()
    n = len(ts)
    idx = np.random.RandomState(0).choice(n, min(n, 8000), replace=False)
    ts, y = ts[idx], y_np[idx]
    d = np.sqrt(((ts[:, None] - ts[None]) ** 2).sum(-1))
    same = (y[:, None] == y[None])
    eye = np.eye(len(y), dtype=bool)
    intra = d[same & ~eye].mean()
    inter = d[~same].mean()
    return intra, inter


def centroid_probe(ts_tr, y_tr, ts_te, y_te):
    """Nearest-centroid device classification in theta_Tx space."""
    mu = np.stack([ts_tr[y_tr == c].mean(0) for c in np.unique(y_tr)])
    d = ((ts_te[:, None] - mu[None]) ** 2).sum(-1)
    return (d.argmin(1) == y_te).mean() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data_cache/wisig_manySig_raw_full.npz')
    ap.add_argument('--folds', default='0,1')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--C', type=int, default=8)
    ap.add_argument('--pdim', type=int, default=32)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--supcon_w', type=float, default=0.5)
    ap.add_argument('--adv_w', type=float, default=0.1)
    ap.add_argument('--recon_w', type=float, default=0.1)
    ap.add_argument('--ortho_w', type=float, default=0.1)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    iq, tx, rx, day = load_data(args.npz)
    X = torch.from_numpy(iq).to(args.device)
    y = torch.from_numpy(tx)
    day_dev = torch.from_numpy(day).to(args.device)
    n_tx = int(tx.max()) + 1
    n_rx = int(rx.max()) + 1
    print(f'[v9f] NPIE-v3 (SupCon+GRL on theta_Tx) | tx={n_tx} rx={n_rx} '
          f'folds={args.folds} seeds={args.seeds} ep={args.epochs} C={args.C} '
          f'supcon_w={args.supcon_w} adv_w={args.adv_w}', flush=True)

    folds = [int(f) for f in args.folds.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    all_acc, all_probe, all_ratio = [], [], []
    for f in folds:
        tr, te = crd_split(rx, day, f)
        tr = torch.from_numpy(tr).long()
        te = torch.from_numpy(te).long()
        for sd in seeds:
            torch.manual_seed(sd)
            bb = BlackBoxStream(C=args.C, L=256, emb_dim=512).to(args.device)
            pTx = ParamNet(512, 256, args.pdim).to(args.device)
            pRx = ParamNet(512, 256, 8).to(args.device)
            clf = Classifier(512 + args.pdim, n_tx, hidden=256).to(args.device)
            recon = ReconNet(args.pdim, 8, 256, 512).to(args.device)
            # SupCon projection head + GRL adversaries
            proj = nn.Sequential(nn.Linear(args.pdim, 64), nn.ReLU(),
                                 nn.Linear(64, 64)).to(args.device)
            adv_rx = nn.Linear(64, n_rx).to(args.device)
            adv_day = nn.Linear(64, 2).to(args.device)

            phys_params = list(pTx.parameters()) + list(pRx.parameters()) + list(recon.parameters())
            params = (list(bb.parameters()) + list(clf.parameters()) + phys_params
                      + list(proj.parameters()) + list(adv_rx.parameters())
                      + list(adv_day.parameters()))
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
            crit = nn.CrossEntropyLoss(label_smoothing=0.1)
            Xtr = X[tr]; ytr = y[tr].to(args.device)
            rxtr = torch.from_numpy(rx[tr.numpy()]).to(args.device)
            daytr = day_dev[tr]
            best = 0
            for ep in range(args.epochs):
                warmup = ep < args.warmup
                lambd = ep / max(args.epochs - 1, 1)   # GRL ramp 0->1
                for p in phys_params:
                    p.requires_grad_(not warmup)
                perm = torch.randperm(Xtr.shape[0])
                Xs, ys, rs, ds = Xtr[perm], ytr[perm], rxtr[perm], daytr[perm]
                for i in range(0, Xtr.shape[0], args.batch):
                    xb, yb = Xs[i:i + args.batch], ys[i:i + args.batch]
                    rb, db = rs[i:i + args.batch], ds[i:i + args.batch]
                    flat = xb.reshape(xb.shape[0], -1)
                    bb_feat = bb(xb)
                    ts = scale_tx(pTx(flat)); rb_p = pRx(flat)
                    fused = torch.cat([bb_feat, ts], -1)
                    L_main = crit(clf(fused), yb)
                    if warmup:
                        loss = L_main
                        L_supcon = L_adv = L_recon = L_ortho = torch.zeros(1, device=args.device)
                    else:
                        zp = proj(ts)
                        L_supcon = supcon_loss(zp, yb)
                        rev = GRL.apply(zp, lambd)
                        L_adv = crit(adv_rx(rev), rb) + crit(adv_day(rev), db)
                        L_recon = F.mse_loss(recon(ts, rb_p), flat) / (flat.var() + 1e-8)
                        L_ortho = ortho_loss(ts, rb_p)
                        loss = (L_main + args.supcon_w * L_supcon + args.adv_w * L_adv
                                + args.recon_w * L_recon + args.ortho_w * L_ortho)
                    opt.zero_grad(); loss.backward(); opt.step()
                sched.step()
                with torch.no_grad():
                    xb = X[te]; yb = y[te].to(args.device)
                    flat = xb.reshape(xb.shape[0], -1)
                    bb_feat = bb(xb)
                    ts = scale_tx(pTx(flat))
                    fused = torch.cat([bb_feat, ts], -1)
                    pred = clf(fused).argmax(1)
                    acc = (pred == yb).float().mean().item() * 100
                if acc > best: best = acc
                wtag = "WARMUP" if warmup else "train"
                print(f'[v9f f{f} s{sd}] ep{ep+1} {wtag} Lmain={L_main.item():.3f} '
                      f'Lsup={L_supcon.item():.3f} Ladv={L_adv.item():.3f} '
                      f'Lrec={L_recon.item():.3f} | val={acc:.2f}% best={best:.2f}%',
                      flush=True)
            # fingerprint quality after this run
            with torch.no_grad():
                ftr = X[tr].reshape(X[tr].shape[0], -1)
                fte = X[te].reshape(X[te].shape[0], -1)
                ts_tr = scale_tx(pTx(ftr)).cpu().numpy()
                ts_te = scale_tx(pTx(fte)).cpu().numpy()
            ytr_np = y[tr].numpy(); yte_np = y[te].numpy()
            intra, inter = cluster_quality(ts_te, yte_np)
            ratio = intra / max(inter, 1e-9)
            probe = centroid_probe(ts_tr, ytr_np, ts_te, yte_np)
            all_acc.append(best); all_probe.append(probe); all_ratio.append(ratio)
            print(f'[v9f f{f} s{sd}] FINGERPRINT intra={intra:.3f} inter={inter:.3f} '
                  f'ratio={ratio:.3f} centroid_probe={probe:.2f}%', flush=True)
    print(f'[V9f] CRD mean={np.mean(all_acc):.2f}% (all={[round(a,2) for a in all_acc]})', flush=True)
    print(f'[V9f] centroid probe mean={np.mean(all_probe):.2f}% '
          f'(all={[round(a,2) for a in all_probe]})', flush=True)
    print(f'[V9f] cluster ratio mean={np.mean(all_ratio):.3f} '
          f'(all={[round(a,3) for a in all_ratio]}) | v9e baseline was 8.53', flush=True)


if __name__ == '__main__':
    main()
