from pricing.monte_carlo import monte_carlo_simulation
from pricing.black_scholes import black_scholes_price
import pytest

def test_mc_matches_bs():
    """
    Test that Monte Carlo simulation matches the Black-Scholes price for the base case.
    Base case: S=100, K=100, T=1, r=0.05, sigma=0.2, N=1_000_000, seed=42

    With N=1_000_000 simulations, the standar error should be 0.0147. 
    The tolerance is set to 0.06, roughly 4 times the standard error.:
    - A correct estimator lands more than 4SW away only ~1 in 16,000 times, so failure points to a real bug, not bad luck
    - Typical bugs shift the price by tens os SE, so they are caught easily.
    The seed is fixed so the test is reproducible
    """
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    N = 1_000_000
    TOL = 0.06
    bs_price = black_scholes_price(S, K, T, r, sigma, 'call')
    mc_price, mc_se = monte_carlo_simulation(S, K, T, r, sigma, N, seed=42)
    assert abs(mc_price - bs_price)<TOL