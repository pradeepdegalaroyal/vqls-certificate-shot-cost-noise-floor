# Method

## Problem family

`A x = b` with `A` a real Pauli sum (transverse-field Ising operator: single-qubit `X` terms with random weights in [0.5, 1.5], nearest-neighbour `ZZ` terms with weights in [0.3, 1.0]) shifted by a multiple of the identity and scaled so the spectral norm is 1 and the condition number equals a requested `kappa` exactly. `b` is a random real product state. Sizes: 3 to 6 qubits. This is the Hamiltonian-resolvent form used in linear-response and Green's-function problems.

## Quantum method: variational solve plus certificate

1. **Variational Quantum Linear Solver (VQLS) baseline (Bravo-Prieto et al., 2023).** Real hardware-efficient ansatz (blocks of `Ry` rotations plus a chain of Controlled-NOT (CNOT) gates, then a final `Ry` layer), global cost `C = 1 - |<b|A|x>|^2 / <x|A^2|x>`, exact gradients by parameter shift, trained with the limited-memory Broyden-Fletcher-Goldfarb-Shanno method (L-BFGS) in noise-free statevector simulation (up to 150 iterations, 5 seeds).
2. **Measurable decomposition.** `beta = <b|A|x>` is a sum of Hadamard-test overlaps; `delta = <x|A^2|x>` is a constant plus Pauli expectation values of commuting products (anticommuting pairs cancel; duplicate products are merged).
3. **Certificate.** The proven bound `eps <= kappa * sqrt(C)`. With a finite-shot estimate `C_hat` and delta-method standard deviation `sigma`, the one-sided 95 % certificate is `eps_cert = kappa * sqrt(C_hat + 1.645 sigma)`.
4. **Shots to certify.** The smallest total shot count `S` such that `C_mean + 2 * 1.645 * sigma(S) <= (eps_t / kappa)^2`, i.e. the certificate is valid at 95 % and is issued at the target with 95 % probability. It is infinite if the noise bias alone exceeds the target. The delta-method formula is checked against binomial shot simulation (`results/monte_carlo_validation.csv`).
5. **Variance-optimal allocation (new component).** Shots are allocated across circuit settings in proportion to `|dC/dmu_k| * sqrt(1 - mu_k^2)` (Neyman allocation), versus the equal-shots baseline.
6. **Noise model.** Each CNOT mixes in a global depolarising fraction `p`, so a circuit with `g` CNOTs scales every non-identity expectation by `(1 - p)^g`. CNOT counts: ansatz `layers*(qubits-1)`; Hadamard-test circuits add a controlled copy of the ansatz and of the `b` preparation (Toffoli = 6 CNOT, controlled-`Ry` = 2 CNOT) and one controlled Pauli per term. All-to-all connectivity, no routing.
7. **Zero-noise extrapolation (ZNE) (Kim et al., 2023 style).** Noise factors `lambda = 1, 3, 5` (gate folding), quadratic Richardson weights `15/8, -5/4, 3/8` applied term by term before the cost is assembled. Shot counts include all three factors.

## Classical baseline

Conjugate gradient (CG) on the sparse matrix, started from zero, run until the normalised iterate reaches the same trace distance as the trained variational state. Cost counted as `2 nnz + 10 n` floating-point operations per iteration.

## Metrics

Validity of the bound, looseness `eps_bound / eps_true`, total shots to certify, infinite-shot floor `kappa * sqrt(C_mean)`, mean ± standard deviation over 5 seeds (instances differ in couplings, `b` and optimiser start), CNOT executions versus CG floating-point operations.

## Sweeps and ablation

- Scaling: `kappa` in {4, 8, 16, 32} at 4 qubits; 3 to 6 qubits at `kappa = 8`; targets `eps_t` in {0.1, 0.2, 0.4, 0.8}.
- Noise sweep: `p` from 0 to 1e-3 (8 values), raw versus ZNE.
- Ablation: ansatz depth 1 to 5 layers at 4 qubits, `kappa = 8` (deeper ansatz lowers `C` but adds CNOTs and therefore noise); allocation (equal versus variance-optimal); mitigation (raw versus ZNE).

## Seeds and compute

Seeds 0 to 4 for every setting. Training uses at most 150 L-BFGS iterations per run on the CPU; the certification analysis is exact moment computation, with binomial Monte Carlo only for validation. Total runtime is reported in the README.
