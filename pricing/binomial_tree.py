import numpy as np

def binomial_price(S, K, T, r, sigma, N, option_type):
    '''
    Calculate the price of an European Option using the binomial CRR with N steps
    The binomial tree approximates the underlying asset's price movements over time.
    It calculates the option price by working backward from the terminal nodes of the tree.
    It integrates the risk-neutral probabilites to determine the expected payoff at each node. 
    Finally, discounts one step at each level of the backward introduction.
    
    Parameters:
    S: Spot price of the underlying asset
    K: Strike price of the option
    T: Time to maturity in years
    r: Annual risk-free interest rate, continuously compounded
    sigma: Annual volatility of the underlying asset ('add in decimal (e.g., 0.2 for 20%)')
    N: Number of steps in the binomial tree
    option_type: Type of the option ('call' or 'put')

    Formed Inputs:
    u: Up factor for the underlying asset price
    d: Down factor for the underlying asset price
    p: Risk-neutral probability of an up move
    1-p: Risk-neutral probability of a down move
    dt: Time step size (T/N)
    V: Option value at each node of the binomial tree

    Returns:
    The price of the European option at the initial node of the binomial tree.
    (Float)

    Assumptions:
    - The underlying asset follows a binomial price model with up and down factors.
    - Up and down factors are constant throughout the tree.
    - The risk-neutral probability of an up move is constant throughout the tree.
    - The volatility of the underlying asset is constant throughout the tree.
    - The underlying asset does not contain dividends (it breaks  the assumption that each step is a multiplicative factor, so when dividends are paid, the price does not follow the simple up and down factors).
    - The option is European, meaning it can only be exercised at maturity.
    - CRR = Cox-Ross-Rubinstein model, which defines the up and down factors as u = exp(sigma * sqrt(dt)) and d = 1/u.
    '''
    dt= T/N
    u= np.exp(sigma*np.sqrt(dt))
    d=1/u
    p= (np.exp(r*dt)-d)/(u-d)
    if p >1 or p<0:
        raise ValueError("Risk-neutral probability p must be between 0 and 1.")
    j= np.arange(N+1)
    stock_prices= S*u**j*d**(N-j)
    if option_type == "call":
        V = np.maximum(stock_prices - K, 0)
    elif option_type == "put":
        V = np.maximum(K-stock_prices, 0)
    else: 
        raise ValueError("Option type must be either 'call' or 'put'.")
    for _ in range (N):
        V_up = V[1:]
        V_down = V[:-1]
        V = np.exp(-r*dt)*(p*V_up + (1-p)*V_down)
    return V[0]