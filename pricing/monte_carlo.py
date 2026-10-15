import numpy as np
from pricing.black_scholes import black_scholes_price
from scipy.stats import norm

def monte_carlo_simulation (S,K,T,r, sigma, N, seed= None):
    """
    Description: develop a monte carlo simulation for pricing options. 
    Call European Option (only).

    Inputs:
    S: spot price of the underlying asset
    K: strike price of the option
    T: time to maturity of the option
    r: risk-free interest rate
    sigma: annualized volatility of the underlying asset
    N: number of simulations
    seed: random seed for reproducibility

    Outputs:
    price: estimated price of the option
    se: standar error of the estimated price
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    S_T = S * np.exp((r-sigma**2/2)*T + sigma*np.sqrt(T)*Z)
    payoff = np.maximum(S_T-K, 0)
    disc = np.exp(-r*T)*payoff
    price = np.mean(disc)
    se = np.std(disc, ddof=1)/np.sqrt(N) 
    return price, se



if __name__ == "__main__":
    mc_price, mc_se = monte_carlo_simulation(100, 100, 1, 0.05, 0.2, 10_000, seed=42)
    BS = black_scholes_price(100, 100, 1, 0.05, 0.2, 'call')
    MC = mc_price
    error= abs(MC-BS)
    error_in_se = error/mc_se
    lower, upper = confidence_interval(MC, mc_se)
    print(BS, MC, mc_se, error, error_in_se)

def confidence_interval(price, se, level=0.95):
    """
    Compute the confidence interval for the estimateed price of Monte Carlo simulation
    Inputs:
    price: estimated price of the option
    se: standard error of the estimated price
    level: confidence level (default is 95%)
    
    Outputs:
    lower: lower bound of the confidence interval
    upper: upper bound of the confidence interval
    """
    z = norm.ppf(1-(1-level)/2)
    lower = price -z*se
    upper = price +z*se
    return lower, upper 

