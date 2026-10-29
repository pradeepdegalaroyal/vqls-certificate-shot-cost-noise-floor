"""Fast tests: instance construction, exact gradients, the kappa bound, the estimator moments."""
import numpy as np

from vqls_cert.ansatz import n_params, statevector
from vqls_cert.certify import analyse, monte_carlo, shots_to_certify
from vqls_cert.classical import cg_flops_to_precision
from vqls_cert.cost import (beta_delta, build_terms, cost_and_grad, exact_cost, term_means, train,
                            trace_distance)
from vqls_cert.problem import make_instance


def test_instance_condition_number_and_norm():
    inst = make_instance(3, 12.0, 0)
    ev = np.linalg.eigvalsh(inst.A)
    assert np.isclose(ev[-1] / ev[0], 12.0)
    assert np.isclose(ev[-1], 1.0)
    assert np.isclose(np.linalg.norm(inst.b), 1.0)


def test_statevector_is_normalised():
    th = np.random.default_rng(0).normal(size=n_params(4, 3))
    assert np.isclose(np.linalg.norm(statevector(th, 4, 3)), 1.0)


def test_gradient_matches_finite_difference():
    inst = make_instance(3, 6.0, 1)
    th = np.random.default_rng(2).normal(size=n_params(3, 2))
    _, g = cost_and_grad(th, inst, 2)
    e = 1e-6
    num = np.array([(cost_and_grad(th + e * np.eye(len(th))[j], inst, 2)[0]
                     - cost_and_grad(th - e * np.eye(len(th))[j], inst, 2)[0]) / (2 * e) for j in range(len(th))])
    assert np.abs(g - num).max() < 1e-7


def test_term_decomposition_reproduces_exact_cost():
    inst = make_instance(3, 5.0, 3)
    ts = build_terms(inst)
    x = statevector(np.random.default_rng(4).normal(size=n_params(3, 2)), 3, 2)
    mu_h, mu_p = term_means(x, inst, ts)
    beta, delta = beta_delta(mu_h, mu_p, ts)
    assert np.isclose(1 - beta ** 2 / delta, exact_cost(x, inst))


def test_kappa_bound_holds_along_training():
    inst = make_instance(3, 8.0, 0)
    res = train(inst, 2, 0, maxiter=30)
    for th in res["history"][::5]:
        x = statevector(th, 3, 2)
        assert trace_distance(x, inst.x_star) <= inst.kappa * np.sqrt(exact_cost(x, inst)) + 1e-9


def test_noise_free_analysis_matches_exact_cost_and_zne_removes_bias():
    inst = make_instance(3, 8.0, 0)
    res = train(inst, 2, 0, maxiter=30)
    x = statevector(res["theta"], 3, 2)
    ts = build_terms(inst)
    mu_h, mu_p = term_means(x, inst, ts)
    c = exact_cost(x, inst)
    assert np.isclose(analyse(mu_h, mu_p, ts, 3, 2, 0.0, False).c_mean, c)
    raw = analyse(mu_h, mu_p, ts, 3, 2, 1e-4, False).c_mean
    zne = analyse(mu_h, mu_p, ts, 3, 2, 1e-4, True).c_mean
    assert abs(zne - c) < abs(raw - c)
    assert shots_to_certify(analyse(mu_h, mu_p, ts, 3, 2, 0.5, False), "uniform", c * 2) == float("inf")


def test_neyman_allocation_never_worse_and_monte_carlo_coverage():
    inst = make_instance(3, 8.0, 0)
    res = train(inst, 2, 0, maxiter=30)
    x = statevector(res["theta"], 3, 2)
    ts = build_terms(inst)
    mu_h, mu_p = term_means(x, inst, ts)
    an = analyse(mu_h, mu_p, ts, 3, 2, 0.0, False)
    assert an.v_neyman <= an.v_uniform + 1e-12
    c = exact_cost(x, inst)
    target = 4 * c + 1e-4
    s = shots_to_certify(an, "uniform", target)
    mc = monte_carlo(mu_h, mu_p, ts, 3, 2, 0.0, False, int(np.ceil(s / an.n_items)), c, target, 1500,
                     np.random.default_rng(0))
    assert mc["coverage"] > 0.9 and 0.88 < mc["issue_rate"] < 1.0


def test_conjugate_gradient_reaches_target():
    inst = make_instance(3, 8.0, 0)
    out = cg_flops_to_precision(inst, 1e-3)
    assert out["eps"] <= 1e-3 and out["iterations"] <= inst.n
