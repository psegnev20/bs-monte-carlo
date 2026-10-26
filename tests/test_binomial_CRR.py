from pricing.binomial_tree import binomial_price
from pricing.black_scholes import black_scholes_price
import numpy as np
import pytest

BS_TOL = 0.01

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
N = 1000

def test_converges_to_black_scholes():
    for option_type in ("call", "put"):
        binomial = binomial_price(S, K, T, r, sigma, N, option_type)
        bs = black_scholes_price(S, K, T, r, sigma, option_type) 
        assert binomial == pytest.approx(bs, abs=BS_TOL)
