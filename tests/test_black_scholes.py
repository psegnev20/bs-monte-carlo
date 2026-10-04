import numpy as np
import pytest
from pricing.black_scholes import black_scholes_price 
@pytest.mark.parametrize("S, K, T, r, sigma", [
     (100, 100, 1, 0.05, 0.2),
])
def test_put_call_parity(S, K, T, r, sigma):
    call_price = black_scholes_price(S, K, T, r, sigma, "call")
    put_price = black_scholes_price(S, K, T, r, sigma, "put")
    right_side = S-K*np.exp(-r*T)
    left_side = call_price - put_price
    assert abs(left_side - right_side) < 1e-6