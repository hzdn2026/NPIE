"""共轭循环谱（conjugate SCF）alpha-profile 提取 —— 论文2 特征核心。

布局与论文1 的 cycproj 完全一致：(N, 2, 51, 23)
  - channel 0: 非共轭 SCF  S_x^alpha(f)
  - channel 1: 共轭   SCF  S_{x,x^*}^alpha(f)
  - 51 个循环频率 alpha  bins, 23 个谱频率 f bins

实现说明（reference 版，CPU/numpy 可跑）：
  - 先用循环自相关函数 CAF R^alpha(tau) 估计，再对 tau 做 FFT 得到 S(alpha,f)。
  - 非共轭: R^alpha(tau) = E[ x(t) x*(t-tau) e^{-j2pi alpha t} ]
  - 共轭  : R^alpha(tau) = E[ x(t) x(t-tau)   e^{-j2pi alpha t} ]  (alpha≈2fc 出峰)
  - 真实训练请直接替换为本仓库论文1 的 cycproj 提取器以保证与已发表结果一致。

仅依赖 numpy，可在无 GPU 环境离线验证特征正确性。
"""
import numpy as np


def _caf_nonconj(x, alphas, taus, fs):
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    S = np.zeros((len(alphas), len(taus)), dtype=np.complex128)
    t_idx = np.arange(N)
    for ia, a in enumerate(alphas):
        phase = np.exp(-1j * 2 * np.pi * a * t_idx / fs)
        xa = x * phase
        for it, tau in enumerate(taus):
            tau = int(tau)
            if tau >= 0:
                num = xa[tau:] * np.conj(x[: N - tau])
            else:
                num = xa[: N + tau] * np.conj(x[-tau:])
            S[ia, it] = np.mean(num)
    return S


def _caf_conj(x, alphas, taus, fs):
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    S = np.zeros((len(alphas), len(taus)), dtype=np.complex128)
    t_idx = np.arange(N)
    for ia, a in enumerate(alphas):
        phase = np.exp(-1j * 2 * np.pi * a * t_idx / fs)
        xa = x * phase
        for it, tau in enumerate(taus):
            tau = int(tau)
            if tau >= 0:
                num = xa[tau:] * x[: N - tau]
            else:
                num = xa[: N + tau] * x[-tau:]
            S[ia, it] = np.mean(num)
    return S


def _caf_to_scf(S, taus, fs):
    # FFT over tau axis -> S(alpha, f), then fftshift
    Sf = np.fft.fftshift(np.fft.fft(S, axis=1), axes=1)
    return Sf


def extract_scf(x, fs=1.0, n_alpha=51, n_freq=23, alpha_max=0.3,
                tau_max=64, n_tau=51):
    """提取 (2, n_alpha, n_freq) 的共轭SCF alpha-profile。

    返回: np.float32 array, shape (2, n_alpha, n_freq)
      channel0 = 非共轭 SCF(绝对值, 归一化)
      channel1 = 共轭   SCF(绝对值, 归一化)
    """
    x = np.asarray(x, dtype=np.complex128)
    alphas = np.linspace(-alpha_max, alpha_max, n_alpha)
    taus = np.linspace(0, tau_max, n_tau).astype(int)

    S_nc = _caf_nonconj(x, alphas, taus, fs)
    S_c = _caf_conj(x, alphas, taus, fs)

    S_nc_f = np.abs(_caf_to_scf(S_nc, taus, fs))
    S_c_f = np.abs(_caf_to_scf(S_c, taus, fs))

    # 谱频率轴下采样到 n_freq 个中心 bins
    n_f = S_nc_f.shape[1]
    center = n_f // 2
    half = n_freq // 2
    lo = center - half
    hi = center + half + (n_freq % 2)
    S_nc_f = S_nc_f[:, lo:hi]
    S_c_f = S_c_f[:, lo:hi]

    # 逐通道归一化（与论文1 预处理一致）
    def _norm(a):
        return (a - a.mean()) / (a.std() + 1e-8)

    S_nc_f = _norm(S_nc_f)
    S_c_f = _norm(S_c_f)
    return np.stack([S_nc_f, S_c_f], axis=0).astype(np.float32)


if __name__ == "__main__":
    # 快速自检：合成 BPSK
    from test_scf import gen_bpsk
    x, fs = gen_bpsk()
    feat = extract_scf(x, fs=fs, alpha_max=0.3)
    print("feature shape:", feat.shape)
