"""WiSig ManySig -> 共轭SCF (N,2,51,23) 预计算。

论文2 数据预处理入口。把官方 ManySig.pkl（或 .mat 兜底）转成论文1 同款
共轭循环谱 alpha-profile，供 baseline 复现与 CycDA-RT 训练使用。

数据组织（官方 compact subset .pkl）：
  dataset['data'][tx][rx][day][eq] : ndarray (N, 256, 2)  N<=1000, 256=时间, 2=I/Q
  eq=0 非均衡(默认, 跨接收机标准) / eq=1 均衡

用法：
  # 仅检查数据格式（数据到位即可，不耗时）
  python precompute_scf.py --src raw_wisig/ManySig --inspect

  # 正式预计算
  python precompute_scf.py --src raw_wisig/ManySig \\
      --out ../data_cache/wisig_manySig_scf.npz --samples_per_pair 100

输出 npz 字段：
  scf : (N,2,51,23) float32   ch0=非共轭SCF, ch1=共轭SCF
  tx  : (N,) int  (0..5       发射机标签)
  rx  : (N,) int  (0..11      接收机标签, 跨接收机域)
  day : (N,) int  (0..3       采集日,   跨时间域)
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from conjugate_scf import extract_scf


def _load_pkl(src, name="ManySig"):
    import pickle
    path = os.path.join(src, name + ".pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"未找到 {path}（请用官方 ManySig.pkl）")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_mat(src, name="ManySig"):
    """兜底：直接读原始 .mat（Tx1.mat..Tx6.mat）。结构以 inspect 为准。"""
    from scipy.io import loadmat
    tx_files = sorted(
        [f for f in os.listdir(src) if f.startswith("Tx") and f.endswith(".mat")])
    if not tx_files:
        raise FileNotFoundError(f"{src} 下既无 ManySig.pkl 也无 Tx*.mat")
    tx_list, rx_list, day_list = [], [], []
    data = {}  # (tx,rx,day) -> (N,256,2) ndarray
    for ti, tf in enumerate(tx_files):
        m = loadmat(os.path.join(src, tf))
        # 变量: IdSig_preamble / IdSig_preamble_eq / IdSig_data / IdSig_data_eq
        key = "IdSig_preamble_eq" if "IdSig_preamble_eq" in m else "IdSig_preamble"
        cell = m[key]  # 通常为 (n_rx, n_day) cell, 每格 (256,2,N) 或 (2,256,N)
        for ri in range(cell.shape[0]):
            for di in range(cell.shape[1]):
                arr = np.asarray(cell[ri, di])
                # 归一化轴序 -> (N,256,2)
                if arr.ndim == 3:
                    if arr.shape[0] == 2:
                        arr = arr.transpose(1, 2, 0)  # (256,N,2)->(N,256,2)? 视实际
                    elif arr.shape[2] == 2:
                        arr = arr.transpose(2, 0, 1)  # (256,2,N)->(N,256,2)
                data[(ti, ri, di)] = arr
        tx_list.append(ti)
    return {"tx_list": tx_list, "rx_list": list(range(max(
        [k[1] for k in data]) + 1)), "capture_date_list": list(
        range(max([k[2] for k in data]) + 1)), "equalized_list": [0],
        "data": _mat_to_nested(data)}


def _mat_to_nested(data):
    ntx = max([k[0] for k in data]) + 1
    nrx = max([k[1] for k in data]) + 1
    nday = max([k[2] for k in data]) + 1
    nested = [[[[data.get((t, r, d), np.zeros((0, 256, 2)))
                 for _ in [0]] for d in range(nday)]
               for r in range(nrx)] for t in range(ntx)]
    return nested


def inspect(src):
    print(f"[inspect] 扫描 {src}")
    pkl = os.path.join(src, "ManySig.pkl")
    if os.path.isfile(pkl):
        ds = _load_pkl(src)
        print("  格式: ManySig.pkl")
        print("  tx_list           :", ds.get("tx_list"))
        print("  rx_list           :", ds.get("rx_list"))
        print("  capture_date_list :", ds.get("capture_date_list"))
        print("  equalized_list    :", ds.get("equalized_list"))
        # 抽样打一条信号的 shape
        t, r, d, e = 0, 0, 0, 0
        arr = np.asarray(ds["data"][t][r][d][e])
        print(f"  data[{t}][{r}][{d}][{e}].shape = {arr.shape}  "
              f"(期望 (N,256,2))")
        print(f"  dtype={arr.dtype}")
        return
    # .mat 兜底
    from scipy.io import loadmat
    tf = next((f for f in os.listdir(src)
               if f.startswith("Tx") and f.endswith(".mat")), None)
    if tf is None:
        print("  !! 未找到 ManySig.pkl 或 Tx*.mat，请确认解压路径")
        return
    m = loadmat(os.path.join(src, tf))
    keys = [k for k in m.keys() if not k.startswith("__")]
    print("  格式: Tx*.mat ; 变量名:", keys)
    key = "IdSig_preamble_eq" if "IdSig_preamble_eq" in m else "IdSig_preamble"
    cell = m[key]
    print(f"  {key}.shape = {cell.shape}")
    arr = np.asarray(cell[0, 0])
    print(f"  cell[0,0].shape = {arr.shape}  (期望 (256,2,N) 或 (2,256,N))")


def _extract_one(x):
    """单条信号的SCF提取（供进程池调用）。x: 复数 ndarray(256,)。"""
    return extract_scf(x, fs=1.0)


def build(src, out, samples_per_pair=100, equalized=0, fmt="auto", jobs=1):
    if fmt == "auto":
        fmt = "mat" if os.path.isfile(os.path.join(src, "ManySig.pkl")) is False else "pkl"
    ds = _load_pkl(src) if fmt == "pkl" else _load_mat(src)

    tx_list = ds["tx_list"]
    rx_list = ds["rx_list"]
    day_list = ds["capture_date_list"]
    eq_i = ds["equalized_list"].index(equalized)

    # 先收集全部任务，再统一并行/串行提取（避免进程池内重复 import 开销）
    tasks = []  # (ti, ri, di, complex256)
    total_pairs = 0
    used_pairs = 0
    for ti, tx in enumerate(tx_list):
        for ri, rx in enumerate(rx_list):
            for di, day in enumerate(day_list):
                arr = np.asarray(ds["data"][ti][ri][di][eq_i])
                if arr.shape[0] == 0:
                    continue
                total_pairs += 1
                take = min(samples_per_pair, arr.shape[0])
                for k in range(take):
                    sig = arr[k]  # (256,2)
                    x = sig[:, 0].astype(np.float64) + 1j * sig[:, 1].astype(np.float64)
                    tasks.append((ti, ri, di, x))
                used_pairs += 1

    xs = [t[3] for t in tasks]
    if jobs and jobs > 1:
        import multiprocessing as mp
        n = min(int(jobs), mp.cpu_count())
        print(f"[build] 并行预计算: {n} 进程, 共 {len(xs)} 条信号")
        with mp.Pool(n) as pool:
            feats = pool.map(_extract_one, xs)
    else:
        print(f"[build] 串行预计算, 共 {len(xs)} 条信号")
        feats = [_extract_one(x) for x in xs]

    scf = np.stack(feats).astype(np.float32)
    meta = dict(scf=scf,
                tx=np.array([t[0] for t in tasks], np.int64),
                rx=np.array([t[1] for t in tasks], np.int64),
                day=np.array([t[2] for t in tasks], np.int64))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, **meta)
    print(f"[done] 样本数 N={scf.shape[0]}  feature={scf.shape[1:]}  "
          f"覆盖(Tx,Rx,Day)对={used_pairs}/{total_pairs}")
    print(f"       保存 -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="raw_wisig/ManySig",
                    help="ManySig.pkl 所在目录(或含 Tx*.mat 的目录)")
    ap.add_argument("--out", default="../data_cache/wisig_manySig_scf.npz")
    ap.add_argument("--samples_per_pair", type=int, default=100,
                    help="每(Tx,Rx,Day)取前多少条信号(官方实验用100)")
    ap.add_argument("--equalized", type=int, default=0, choices=[0, 1],
                    help="0=非均衡(默认), 1=均衡")
    ap.add_argument("--fmt", default="auto", choices=["auto", "pkl", "mat"])
    ap.add_argument("--jobs", type=int, default=1,
                    help="SCF预计算进程数(>1 多进程并行, 多核服务器建议32)")
    ap.add_argument("--inspect", action="store_true",
                    help="只检查数据格式, 不计算SCF")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.src)
    else:
        build(args.src, args.out, args.samples_per_pair, args.equalized,
              args.fmt, jobs=args.jobs)
