# Research gap: what does it cost to *certify* a variational quantum linear solve?

## State of the art (from the supplied literature)

The brief tags every paper "other", so the roles below are my own classification by date, citation count and article type.

**Variational linear algebra is the accepted near-term route.** Bravo-Prieto et al., 2023 (doi:10.22331/q-2023-11-22-1188) introduce the Variational Quantum Linear Solver (VQLS). It minimises a cost `C` built from a Hadamard-test estimate of `<b|A|x>`, and it proves the termination condition `C >= eps^2 / kappa^2`, where `eps` is the trace distance to the exact solution and `kappa` the condition number of `A`. That inequality is the only precision guarantee a user of a variational solve gets. Recent application papers use variational or hybrid solves inside scientific codes: Kouskiya and Oskay, 2027 (doi:10.1016/j.cma.2026.119443) for nonlinear finite elements, and Ying et al., 2027 (doi:10.1016/j.jocs.2026.103048) for a hybrid lattice Boltzmann transport model, where the sampling error follows the expected inverse-square-root law in the number of shots.

**Noise is handled by error mitigation, shown for expectation values.** Kim et al., 2023 (doi:10.1038/s41567-022-01914-3) show zero-noise extrapolation (ZNE) at 26 qubits for expectation values. The VQLS cost is not an expectation value: it is `1 - beta^2 / delta`, a ratio of two noisy estimates that nearly cancel at the optimum.

**The reviews ask for cost accounting.** Peral-Garcia et al., 2024 (doi:10.1016/j.cosrev.2024.100619) note that current devices are not yet fault tolerant. Huynh et al., 2027 (doi:10.1016/j.cosrev.2026.101072) list scalability and data embedding as open questions. Alzboun and Abed-alguni, 2027 (doi:10.1016/j.ins.2026.124009) state that the reasons variational agents fail to beat classical methods "remain poorly understood". Jerbi et al., 2023 (doi:10.1038/s41467-023-36159-y) show that model class decides resource scaling.

### Related work table

| Paper | Year | Role (my label) | What it does | What it does NOT cover |
|---|---|---|---|---|
| Bravo-Prieto et al. (doi:10.22331/q-2023-11-22-1188) | 2023 | landmark, baseline | VQLS, Hadamard-test cost, bound `C >= eps^2/kappa^2` | shots needed to *evaluate* the bound; any noise model for the bound |
| Kim et al. (doi:10.1038/s41567-022-01914-3) | 2023 | landmark | ZNE for expectation values up to 26 qubits | mitigation of a nonlinear ratio cost; variance inflation vs certification |
| Jerbi et al. (doi:10.1038/s41467-023-36159-y) | 2023 | landmark | linear quantum models vs re-uploading: qubit and data scaling | linear-algebra tasks; measurement cost of a guarantee |
| Bauer et al. (doi:10.1103/PRXQuantum.4.027001) | 2023 | review, roadmap | quantum simulation roadmap for high-energy physics | cost of verifying variational outputs |
| Bluvstein et al. (doi:10.1038/s41586-023-06927-3) | 2024 | landmark, hardware | 280-physical-qubit logical processor | variational workloads and their verification |
| King et al. (doi:10.1038/s41586-023-05867-2) | 2023 | adjacent | 5000-qubit annealer, scaling exponents vs Monte Carlo | gate-model linear solves |
| Koh et al. (doi:10.1038/s41567-023-02076-6) | 2023 | adjacent | 14-qubit measurement-induced transition on hardware | solver certification |
| Peral-Garcia et al. (doi:10.1016/j.cosrev.2024.100619) | 2024 | review | systematic review of quantum machine learning (QML) | quantitative shot budgets |
| Huynh et al. (doi:10.1016/j.cosrev.2026.101072) | 2027 | review | survey of quantum-inspired learning, open problems | variational-solver verification |
| Alzboun and Abed-alguni (doi:10.1016/j.ins.2026.124009) | 2027 | recent | explainability framework for why quantum agents fail | linear algebra; measurement cost |
| Kouskiya and Oskay (doi:10.1016/j.cma.2026.119443) | 2027 | recent | variational solver for nonlinear elasticity, 1D | shot cost of trusting the result |
| Ying et al. (doi:10.1016/j.jocs.2026.103048) | 2027 | recent, adjacent | hybrid lattice Boltzmann with hardware sampling, shot-error law | solver-level certificate |
| Shirakawa et al. (doi:10.1016/j.future.2026.108731) | 2027 | recent | closed-loop quantum plus supercomputer electronic structure | cost accounting of a verified linear solve |

## The gap

Every application paper above reports *accuracy*, and the one paper that gives a guarantee (VQLS) states it for exact cost values. Nobody in the supplied set quantifies the number of shots, or the hardware noise level, at which that guarantee can actually be *issued*. This matters because near the optimum `C` is tiny (here about 1e-5 to 1e-4) while the shot noise of its estimator does not shrink with `C`, so the relative precision needed grows as the solution improves.

**Hypotheses (falsifiable).**

- **H1 (validity).** `kappa * sqrt(C)` never underestimates the true trace distance along the optimisation. Falsified by one violation.
- **H2 (shot scaling).** To issue a 95 % certificate for a fixed target `eps_t` with 95 % probability, the total shots scale roughly like `kappa^4` in the noise-free case (the variance of the cost estimator divided by the squared gap `eps_t^2/kappa^2 - C`). Falsified if the fitted log-log slope against `kappa` is below 3.
- **H3 (noise floor).** Under depolarising noise per Controlled-NOT (CNOT) gate, the raw certificate has an infinite-shot floor that exceeds a modest target (`eps_t = 0.2`) at error rates where ZNE still certifies; ZNE moves the tolerable error rate up by at least a factor of 10. Falsified otherwise.
- **H4 (no advantage).** At these sizes, certification alone costs several orders of magnitude more operations than the classical conjugate-gradient (CG) solve that produces the answer.

**Success metric.** Total shots to certify (95 % confidence, issued with 95 % probability), the infinite-shot certified-precision floor, and the ratio of CNOT executions to classical floating-point operations.

**Scale and honesty.** Everything is statevector or exact-moment simulation at 3 to 6 qubits (matrix sizes 8 to 64). Hardware statements are model-based. No quantum advantage is claimed.

**Novelty in one sentence:** this study measures the shot cost and the noise floor of *issuing* the VQLS precision certificate, and tests whether variance-optimal shot allocation and ZNE make it affordable.
