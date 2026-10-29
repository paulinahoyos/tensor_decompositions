# staralg

Singular value decompositions and SVD-based tensor decompositions (HOSVD, ST-HOSVD,
TT-SVD, TTN-SVD) for matrices and tensors whose entries lie in a finite-dimensional
\*-algebra K. This is the code for the paper

> Paulina Hoyos and Joe Kileel, *Singular value decompositions over \*-algebras with
> applications to tensor decompositions*, 2026.

The repository has two parts:

- the library `staralg`, which implements the algorithms of the paper;
- the scripts that reproduce the numerical examples of Section 6.

## Installation

Python 3.10 or later is required; 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # library + pytest
pip install -e ".[all]"          # also the data (ERA5) and plotting dependencies
pytest -q                        # run the tests
```

The figures are typeset with LaTeX (matplotlib `text.usetex`, fonts `newtxtext` and
`newtxmath`), so plotting needs a TeX installation.

## Reproducing the numerical examples

Every experiment has three steps: generate or download its data into `data/processed/`
(not tracked), run the experiment, which writes a CSV to `results/`, and plot the figure
from that CSV into `figures/`. The CSV files and figures of the paper are committed, so
you can redraw the figures without rerunning the experiments.

| Paper | Data | K | Script | Results | Figure |
|---|---|---|---|---|---|
| Fig. 5, §6.1 | Noisy three-qubit quantum circuits, 500 x 500 matrix over K | M_8(C) | `exp10_circuits_svd.py` | `exp10_noise0.csv` | `exp10_noise0_paper_log` |
| Fig. 6, §6.2 | Vlasov–Poisson, 48 x 48 x 32 tensor over K | R[Z_32 x Z_32] | `exp3_vlasov.py` | `exp3_format.csv` | `exp3_format_paper_log` |
| Fig. 7, §6.3 | ERA5 2019 anomalies, order-4 tensor over K | R[Z_64] | `exp2_era5.py --format` | `exp2_lon_format.csv` | `exp2_lon_format_paper_log` |
| Fig. 8, §6.3 | ERA5 2019 anomalies, order-3 tensor over K | R[Z_64] (x) DCT_32 | `exp2_era5.py --format` | `exp2_lonlat_format.csv` | `exp2_lonlat_format_paper_log` |
| §6.3, tree tensor networks | ERA5 2019 anomalies, order-4 tensor over K | R[Z_64] | `exp2_era5.py --ttn` | `exp2_lon_ttn.csv` | `exp2_lon_ttn_paper`, `exp2_lon_ttn_paper_log` |

The run times below were measured on a MacBook Air (M2, 8 GB of RAM).

```bash
pip install -e ".[all]"

# Figure 5: quantum circuits (simulated, about a minute, 128 MB)
python data/simulate_circuits.py
python experiments/exp10_circuits_svd.py          # a few minutes
python experiments/plot_paper.py exp10_noise0 --logx --logy --xmax 0.1 --ymax 2000 --mark-gap 0.04 \
    --out figures/exp10_noise0_paper_log

# Figure 6: Vlasov-Poisson (simulated, a few minutes, 300 MB)
python data/simulate_vlasov.py
python experiments/exp3_vlasov.py                 # about 20 minutes
python experiments/plot_paper.py exp3_format --xmax 0.5 --xstep 0.1 --logy --legend "lower right" --ncol 2 \
    --drop "TT over K" --out figures/exp3_format_paper_log

# Figures 7 and 8, and the tree tensor networks: ERA5
# (WeatherBench 2, anonymous Google Cloud Storage; about 1.3 GB transferred, 200 MB kept)
python data/fetch_era5.py
python experiments/exp2_era5.py                   # --format (about 30 minutes), then --ttn (about 7 minutes)
python experiments/plot_paper.py exp2_lon_format --xmax 0.5 --xstep 0.1 --logy --out figures/exp2_lon_format_paper_log
python experiments/plot_paper.py exp2_lonlat_format --xmax 0.5 --xstep 0.1 --logy --ymax 100 \
    --out figures/exp2_lonlat_format_paper_log
python experiments/plot_paper.py exp2_lon_ttn --xmax 0.5 --xstep 0.1 --out figures/exp2_lon_ttn_paper
python experiments/plot_paper.py exp2_lon_ttn --xmax 0.5 --xstep 0.1 --logy --out figures/exp2_lon_ttn_paper_log
```

`exp3_vlasov.py` and `exp2_era5.py` take `--quick` (every fourth tolerance) for a first
look. Each experiment script prints a table of relative errors at fixed compression ratios.
To print that table again from a results file, run
`python experiments/common.py results/<name>.csv`.

Every CSV row has a method, a truncation, the number of real parameters stored, the
compression ratio (real size of the data divided by that number, with a complex number
counted as two reals) and the relative error ||A - A_hat|| / ||A||.

## Repository layout

```
staralg/algebras/   *-algebras and their structure isomorphisms
staralg/core/       svd_K, llsv_K, truncation; HOSVD, ST-HOSVD (tucker.py, tucker_nc.py);
                    TT-SVD (tt.py); TTN-SVD (ttn.py)
staralg/metrics.py  relative error, storage counts, compression ratio
data/               scripts that simulate or download the data
experiments/        one script per experiment, and plot_paper.py for the figures
results/            CSV files of the paper's experiments
figures/            the paper's figures (PDF and PNG)
tests/              one test file per module
```

## Conventions

**Coefficient form.** Data are ordinary real or complex arrays. You choose which axes hold
the algebra's coordinates (`alg_axes`); the remaining axes are the modes of the matrix or
tensor over K. Inside the library, the algebra axes are moved to the end
(`move_alg_axes_last`), so an element of K has shape `(..., *K.shape)`. Errors are always
measured in coefficient form.

**Block form (internal).** `K.forward(x)` applies the structure isomorphism
pi: K -> M_{d_1}(D_1) (+) ... (+) M_{d_l}(D_l) and returns a list of `Group`s, each holding
blocks of one size and type with shape `(..., B, m, m)`. Quaternion blocks are stored in
their complex realization, entry by entry
(a + bi + cj + dk -> [[a+bi, c+di], [-c+di, a-bi]]), so m = 2d for D = H.
`K.inverse(groups)` maps back. Each block has a weight c_rho with
||x||^2 = sum_rho c_rho ||pi_rho(x)||_F^2. The DFT structure map is the unnormalized FFT,
which is multiplicative, so its weights are 1/n (2/n for conjugate pairs stored once).

## Algebras

| Class | Algebra | Blocks |
|---|---|---|
| `Trivial(field)` | R or C (classical case) | one 1x1 block |
| `TubeDFT(n, field)` | F[Z_n], the t-product | DFT; for F = R, conjugate pairs stored once |
| `AbelianGroupAlgebra(n1, ..., nk, field)` | F[Z_n1 x ... x Z_nk] (any finite abelian group), coordinates on k axes | k-dim DFT, one 1x1 block per wavevector; conjugate pairs stored once |
| `GroupAlgebra(table, shape, field)` | F[G] for any finite group G, from its multiplication table | one block per irrep, computed numerically; real, complex or quaternionic by the Frobenius-Schur indicator |
| `GroupAlgebra.dihedral(n)`, `.quaternion()`, `.symmetric(n)`, `.product(K1, K2, ...)` | D_n, Q_8, S_n, products of groups | as above |
| `StarM(M)` | R^n with the *_M product, M orthogonal | n real 1x1 blocks |
| `StarM.from_data(X, axis)` | *_M with a data-driven M | n real 1x1 blocks |
| `TubeDCT(n)` | *_M with the orthonormal DCT-II | n real 1x1 blocks |
| `WalshHadamard(n)` | R[(Z_2)^k], n = 2^k | n real 1x1 blocks |
| `RandomOrthogonal(n, seed)` | *_M with a random orthogonal M | n real 1x1 blocks |
| `Cl10()` | split-complex numbers ~ R (+) R | blocks u + b, u - b |
| `Cl03()` | Cl_{0,3} ~ H (+) H | two quaternion blocks |
| `Cl30()` | Cl_{3,0} ~ M_2(C) (Pauli) | one block |
| `Quaternions()` | H = Cl_{0,2}, basis (1, i, j, k) | one quaternion block |
| `Cl04()` | Cl_{0,4} ~ M_2(H) | one block M_2(H) |
| `MatrixAlgebra(d, D)` | M_d(D), D in {R, C, H} | one block |
| `DirectSum(K1, K2, ...)` | K1 (+) K2 (+) ..., e.g. M_2(C) (+) M_2(C) | the blocks of the summands |
| `TensorProduct(*factors)` | products with at most one non-*_M factor; a product of group algebras is returned as the group algebra of the product group (`TensorProduct(TubeDFT(32), TubeDFT(32))` is `AbelianGroupAlgebra(32, 32)`) | products of blocks; for groups, the irreps of the product group |

Clifford blades are in grade order, e.g. (1, e1, e2, e3, e12, e13, e23, e123), and the
involution is e_k^* = e_k^{-1}. `K.describe()` prints the blocks and weights of an algebra.

## Usage

### SVD over K

```python
import numpy as np
from staralg import MatrixAlgebra, svd_K, rel_error

K = MatrixAlgebra(8, "C")                     # K = M_8(C), coordinates: the 64 matrix entries
A = np.random.randn(50, 40, 64) + 1j * np.random.randn(50, 40, 64)   # 50 x 40 matrix over K
svd = svd_K(A, K)                             # Algorithm 1, blockwise
A4 = svd.reconstruct(rank=4)                  # K-rank 4 truncation (optimal, Theorem 4.2)
Ab = svd.reconstruct(budget=svd.n_params(rank=4))   # per-block ranks, same storage
U, S, V = svd.factors(4)                      # factors over K, in coefficient form
rel_error(A, A4), svd.n_params(rank=4)
```

If the algebra axes are not last: `svd_K(A, K, alg_axes=0)`.

### Tucker, tensor train and tree tensor networks over K

```python
import numpy as np
from staralg import AbelianGroupAlgebra, Cl03, TubeDFT, hosvd_K, sthosvd_K, tt_svd_K, ttn_svd_K

# Tucker: commutative K (raises NoncommutativeTuckerError otherwise)
X = np.random.randn(64, 32, 13, 5, 30)            # lon, lat, level, var, time
K = TubeDFT(64)                                   # longitude in the algebra: alg_axes=0
h = hosvd_K(X, K, alg_axes=0)                     # mode SVDs once
T = h.truncate(ranks=(8, 6, 3, 20))               # uniform K-ranks
T = h.truncate(eps=0.1, adaptive=True)            # per-block ranks, error <= 0.1 ||X||
T = sthosvd_K(X, K, eps=0.1, alg_axes=0, order=(3, 0, 1, 2))   # sequentially truncated
X_hat, storage = T.reconstruct(), T.n_params()

# Tensor train: any K
K = Cl03()
Y = np.random.randn(6, 8, 8, 8, 8)                # an order-4 tensor over Cl(0,3)
tt = tt_svd_K(Y, K, ranks=(2, 4, 2))              # uniform K-ranks
tt = tt_svd_K(Y, K, eps=0.2, adaptive=True, order=(3, 0, 1, 2))
tt.reconstruct(), tt.n_params(), tt.tails, tt.ranks_K

# Tree tensor network (TTN-SVD, Algorithm 6): commutative K
K = AbelianGroupAlgebra(16, 16)
Z = np.random.randn(16, 16, 10, 12, 8, 9)         # the first two axes hold the algebra
tree = (((0,), (1, 2)), 3)                        # vertices are tuples, ints are the modes
ttn = ttn_svd_K(Z, K, tree, eps=0.2, adaptive=True, alg_axes=(0, 1))
ttn.reconstruct(), ttn.n_params(), ttn.tails
```

Error guarantees, with tau_k the weighted energy discarded at mode, step or vertex k:

| Method | Algebras | Error |
|---|---|---|
| HOSVD | commutative K | tau_k <= err^2 <= sum_k tau_k |
| ST-HOSVD | commutative K | err^2 = sum_k tau_k |
| TT-SVD | any K | err^2 = sum_k tau_k, and tau_k <= best rank-r_k error^2 of unfolding k |
| TTN-SVD | commutative K | err^2 = sum_v tau_v, and tau_v <= best rank-r_v error^2 of the unfolding along the edge of v |

With `eps`, every mode or step may discard eps^2 ||X||^2 / N (Tucker) or divided by the
number of steps (TT, TTN), so err <= eps ||X||. `adaptive=True` chooses the ranks per block.

Tucker over a noncommutative K is available with `allow_noncommutative=True`
(`staralg/core/tucker_nc.py`). Mode products then act from the left. For every K,
||X - X_hat||^2 is the sum of the energies removed while forming the core. For the HOSVD
over a noncommutative K this sum is not bounded by the tails of X, and the result depends on
the choice of singular vectors (counterexamples in `tests/test_tucker_nc.py`).

The classical decompositions of the same data are the same functions with `Trivial("R")` or
`Trivial("C")` and no `alg_axes`: the algebra coordinates then form an ordinary mode.

## Citation

```bibtex
@misc{hoyos_kileel_2026,
  author = {Hoyos, Paulina and Kileel, Joe},
  title  = {Singular value decompositions over *-algebras with applications to tensor decompositions},
  year   = {2026},
}
```

## License

MIT; see `LICENSE`.
