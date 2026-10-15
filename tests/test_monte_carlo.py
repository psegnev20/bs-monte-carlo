from pricing.monte_carlo import monte_carlo_simulation
from pricing.black_scholes import black_scholes_price
from pricing.monte_carlo import confidence_interval
import pytest

def test_mc_matches_bs():
    """
    Test that Monte Carlo simulation matches the Black-Scholes price for the base case.
    Base case: S=100, K=100, T=1, r=0.05, sigma=0.2, N=1_000_000, seed=42

    With N=1_000_000 simulations, the standard error is about 0.0147 
    The tolerance is set to 0.06, roughly 4 times the standard error:
    - A correct estimator lands more than 4SE away only ~1 in 16,000 times, so failure points to a real bug, not bad luck
    - Typical bugs shift the price by tens of SE, so they are caught easily.
    The seed is fixed so the test is reproducible
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    N = 1_000_000
    TOL = 0.06
    bs_price = black_scholes_price(S, K, T, r, sigma, 'call')
    mc_price, mc_se = monte_carlo_simulation(S, K, T, r, sigma, N, seed=42)
    error = abs(mc_price - bs_price)
    assert error < TOL, f"MC={mc_price:.4f}, BS={bs_price:.4f}, error={error:.4f} ({error/mc_se:.1f} SE)"

def test_ci_coverage():
    """
    Test that that the 95% confidence interval of the Monte Carlo contains the correct average
    Coverage is the fraction of confidence intervals that contain the true price (Black-Scholes) when the Monte Carlo is repeated with different seeds.
    A correct 95% interval should cover the true price about 95% of the tim Implementation:
    - Each Monte Carlo run uses 10_000 simulations, enough for the Central Limit Theorem to apply, and cheap enough to repeat
    - Monte Carlo simulations are repeated 400 times, providing a precise and faster estimation of the coverage (in just seconds)
    - The measured coverage is itself noisy: its standard deviation is
    sqrt(0.95 * 0.05 / 400) ≈ 0.011. The test accepts coverage in
    [0.91, 0.99], about 0.95 ± 3.5 standard deviations: a correct interval
    falls outside only ~1 in 2,000 times, while a badly built interval
    (e.g. z = 1 gives ~0.68) is caught.
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    N=10_000
    M=400
    bs_price = black_scholes_price(S, K, T, r, sigma, 'call')
    hits = 0
    for seed in range(M):
        mc_price, mc_se = monte_carlo_simulation(S, K, T, r, sigma, N, seed=seed)
        lower, upper = confidence_interval(mc_price, mc_se, level=0.95)
        if lower <= bs_price <= upper:
            hits += 1
    coverage = hits / M
    assert 0.91 <= coverage <= 0.99, f"Coverage={coverage:.3f}"

        