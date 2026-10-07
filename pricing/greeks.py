from pricing.black_scholes import black_scholes_price
import numpy as np
import scipy.stats as st


def fd_delta(S, K, T, r, sigma, option_type, h=None):
    if h is None:
        h = 0.0001*S
        S_up = S + h 
        S_down = S-h
        nudge = black_scholes_price(S_up, K, T, r, sigma, option_type) - black_scholes_price(S_down, K, T, r, sigma, option_type)
        return nudge/(2*h)

def delta(S, K, T, r, sigma, option_type):
    d1 = (np.log(S/K) + (r+sigma**2/2)*T) / (sigma*np.sqrt(T))
    Nd1= st.norm.cdf(d1)
    option_type = option_type.lower()
    if option_type == "call":
        return Nd1
    elif option_type == "put":
        return Nd1 - 1
    else: 
        raise ValueError("Invalid option type. Please specify 'call' or 'put'.")