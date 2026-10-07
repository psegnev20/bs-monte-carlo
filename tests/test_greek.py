import pytest
from pricing.greeks import delta, fd_delta

@pytest.mark.parametrize("S, K, T, r, sigma", [
     (100, 100, 1, 0.05, 0.2),
     (120, 100, 1, 0.05, 0.2),
     (80, 100, 1, 0.05, 0.2),
     (100, 100, 0.1, 0.05, 0.2),
     (100, 100, 1, 0.05, 0.5),

])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_fd(S, K, T, r, sigma, option_type):
   assert delta(S, K, T, r, sigma, option_type) == pytest.approx(fd_delta(S, K, T, r, sigma, option_type), rel=1e-5, abs=1e-8)

