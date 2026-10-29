"""2D2V Vlasov-Poisson simulation (nonlinear Landau damping) for experiment 3.

Equations (electrons, normalized units, neutralizing ion background):
    d_t f + v . grad_x f - E . grad_v f = 0,     div E = 1 - int f dv,   E = -grad phi,
on x, y in [0, 2 pi / k) (periodic) and v_x, v_y in [-v_max, v_max).

Initial condition (nonlinear Landau damping):
    f0 = exp(-|v|^2 / 2) / (2 pi) * (1 + alpha cos(k x) + alpha cos(k y)),  alpha = 0.5, k = 0.5.

Solver: Strang splitting with spectral (FFT) shifts, exact for the periodic directions:
x-advection (dt/2), y-advection (dt/2), velocity advection by E (dt), y (dt/2), x (dt/2).
The velocity domain is treated as periodic; f is about 1e-8 at |v| = 6.

Output: data/processed/vlasov.npz with
    f   (x 32, y 32, vx 48, vy 48, t 32) float32: snapshots at t = 1.25, 2.5, ..., 40
    x, y, vx, vy, t, and the electric energy history (t_hist, W_E)
The velocity grid (48 points, dv = 0.25) puts the recurrence time 2 pi / (k dv) at 50,
after the last snapshot.  About 300 MB.

Usage:
    python data/simulate_vlasov.py               # a few minutes on a laptop
    python data/simulate_vlasov.py --nv 32 --tmax 20 --nsnap 16   # quick check
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def simulate(nx=32, nv=48, vmax=6.0, k=0.5, alpha=0.5, dt=0.1, tmax=40.0, nsnap=32, verbose=True):
    L = 2 * np.pi / k
    x = np.arange(nx) * L / nx
    v = -vmax + np.arange(nv) * (2 * vmax / nv)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=L / nx)          # wavenumbers in x (and y)
    xi = 2 * np.pi * np.fft.fftfreq(nv, d=2 * vmax / nv)    # wavenumbers in v

    X, Y, VX, VY = np.meshgrid(x, x, v, v, indexing="ij")
    f = np.exp(-(VX ** 2 + VY ** 2) / 2) / (2 * np.pi) * (1 + alpha * np.cos(k * X) + alpha * np.cos(k * Y))
    del X, Y, VX, VY
    dv2 = (2 * vmax / nv) ** 2
    mass0 = f.sum() * dv2 * (L / nx) ** 2

    # phase factors for the spatial advections over half a step
    sx = np.exp(-1j * kx[:, None, None, None] * v[None, None, :, None] * dt / 2)   # axis 0 with v_x
    sy = np.exp(-1j * kx[None, :, None, None] * v[None, None, None, :] * dt / 2)   # axis 1 with v_y
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0

    def efield(f):
        n = f.sum(axis=(2, 3)) * dv2
        nh = np.fft.fft2(n)
        nh[0, 0] = 0.0
        # div E = 1 - n  ->  i k . E_hat = -n_hat,  E = -grad phi  ->  E_hat = i k n_hat / |k|^2
        Ex = np.real(np.fft.ifft2(1j * KX * nh / K2))
        Ey = np.real(np.fft.ifft2(1j * KY * nh / K2))
        return Ex, Ey

    def energy(Ex, Ey):
        return 0.5 * np.sum(Ex ** 2 + Ey ** 2) * (L / nx) ** 2

    def adv_x(f):
        return np.real(np.fft.ifft(np.fft.fft(f, axis=0) * sx, axis=0))

    def adv_y(f):
        return np.real(np.fft.ifft(np.fft.fft(f, axis=1) * sy, axis=1))

    def adv_v(f, Ex, Ey):
        # d_t f = E . grad_v f  ->  f(v) <- f(v + E dt)
        fh = np.fft.fft2(f, axes=(2, 3))
        ph = np.exp(1j * (xi[None, None, :, None] * Ex[:, :, None, None]
                          + xi[None, None, None, :] * Ey[:, :, None, None]) * dt)
        return np.real(np.fft.ifft2(fh * ph, axes=(2, 3)))

    nsteps = int(round(tmax / dt))
    snap_steps = set(int(round(j * nsteps / nsnap)) for j in range(1, nsnap + 1))   # t = tmax j / nsnap
    snaps, t_snap, t_hist, W = [], [], [], []
    Ex, Ey = efield(f)
    t_hist.append(0.0)
    W.append(energy(Ex, Ey))
    t0 = time.time()
    for step in range(1, nsteps + 1):
        f = adv_y(adv_x(f))
        Ex, Ey = efield(f)
        f = adv_v(f, Ex, Ey)
        f = adv_x(adv_y(f))
        Ex, Ey = efield(f)
        t_hist.append(step * dt)
        W.append(energy(Ex, Ey))
        if step in snap_steps:
            snaps.append(f.astype(np.float32))
            t_snap.append(step * dt)
            if verbose:
                mass = f.sum() * dv2 * (L / nx) ** 2
                print(f"t = {step * dt:6.2f}   electric energy {W[-1]:.3e}   "
                      f"mass drift {abs(mass - mass0) / mass0:.1e}   ({time.time() - t0:.0f} s)")
    F = np.stack(snaps, axis=-1)                    # (x, y, vx, vy, t)
    return dict(f=F, x=x, y=x, vx=v, vy=v, t=np.array(t_snap), t_hist=np.array(t_hist), W_E=np.array(W))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--nv", type=int, default=48)
    ap.add_argument("--vmax", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--tmax", type=float, default=40.0)
    ap.add_argument("--nsnap", type=int, default=32)
    ap.add_argument("--out", default=str(HERE / "processed" / "vlasov.npz"))
    args = ap.parse_args()
    out = simulate(args.nx, args.nv, args.vmax, dt=args.dt, tmax=args.tmax, nsnap=args.nsnap)
    # linear Landau damping rate for k = 0.5 is about -0.153 (field amplitude), i.e.
    # about -0.306 for the energy; print the early-time fit as a check
    th, lw = out["t_hist"], np.log(out["W_E"])
    peaks = [i for i in range(1, len(lw) - 1) if lw[i] > lw[i - 1] and lw[i] > lw[i + 1] and th[i] < 12]
    if len(peaks) >= 2:
        rate = np.polyfit(th[peaks], lw[peaks], 1)[0]
        print(f"early-time decay rate of the electric energy peaks: {rate:.3f} (linear theory "
              f"-0.307 for small alpha; with alpha = 0.5 nonlinear effects change it)")
    marks = [s for s in (0, 10, 20, 30, 40) if s <= th[-1]]
    print(f"electric energy at t = {marks}: "
          f"{[float(f'{np.exp(lw[np.argmin(abs(th - s))]):.3e}') for s in marks]}")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    print(f"saved {p}: f {out['f'].shape} float32, {out['f'].nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
