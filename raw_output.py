import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price for European options.

    Parameters:
    - S : float : initial stock price
    - K : float : strike price of the option
    - T : float : time to maturity (in years)
    - r : float : risk-free interest rate (annual)
    - sigma : float : volatility of the stock (standard deviation of the stock's returns)
    - option_type : str : type of option ("call" or "put")

    Returns:
    - float : price of the option
    """
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        # Calculate call option price
        call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        return call_price
    elif option_type == 'put':
        # Calculate put option price
        put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        return put_price
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

# Usage example
call_price = black_scholes(100, 100, 1, 0.05, 0.2, 'call')
put_price = black_scholes(100, 100, 1, 0.05, 0.2, 'put')
print(f"Call Price: {call_price:.2f}, Put Price: {put_price:.2f}")
