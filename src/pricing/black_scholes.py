import numpy as np
from scipy.stats import norm


class BlackScholesOptionPricer:
    
    @staticmethod
    def d1(S, K, T, r, sigma, q=0.0):
        if T <= 0 or sigma <= 0:
            return np.nan
        return (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def d2(d1, T, sigma):
        if T <= 0 or sigma <= 0:
            return np.nan
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma, q=0.0):
        d1 = BlackScholesOptionPricer.d1(S, K, T, r, sigma, q)
        d2 = BlackScholesOptionPricer.d2(d1, T, sigma)
        call = S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        return call
    
    @staticmethod
    def put_price(S, K, T, r, sigma, q=0.0):
        call = BlackScholesOptionPricer.call_price(S, K, T, r, sigma, q)
        forward = S * np.exp((r - q) * T)
        discount = np.exp(-r * T)
        put = call - (forward - K) * discount
        return put
    
    @staticmethod
    def implied_vol_from_price(price, S, K, T, r, option_type='call', q=0.0, 
                               initial_guess=0.2, max_iterations=100, tol=1e-6):
        sigma = initial_guess
        
        for _ in range(max_iterations):
            if option_type == 'call':
                price_calc = BlackScholesOptionPricer.call_price(S, K, T, r, sigma, q)
            else:
                price_calc = BlackScholesOptionPricer.put_price(S, K, T, r, sigma, q)
            
            diff = price_calc - price
            if abs(diff) < tol:
                return sigma
            
            #vega
            d1 = BlackScholesOptionPricer.d1(S, K, T, r, sigma, q)
            vega = S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)
            
            if vega < 1e-10:
                break
            
            sigma = sigma - diff / vega
            sigma = max(0.001, min(2.0, sigma))
        
        return sigma
