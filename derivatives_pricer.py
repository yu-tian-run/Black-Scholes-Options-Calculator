
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict


@dataclass
class OptionInputs:
    # Input parameters for option pricing
    spot_price: float      # Current stock price (S)
    strike_price: float    # Strike price (K)
    time_to_expiry: float  # Time to expiration in years (T)
    volatility: float      # Annualized volatility (σ)
    risk_free_rate: float  # Risk-free interest rate (r)
    option_type: str       # 'call' or 'put'
    dividend_yield: float = 0.0  # Dividend yield (q)


class BlackScholesCalculator:
    # Black-Scholes-Merton option pricing model with Greeks calculation
    
    def __init__(self, inputs: OptionInputs):
        # Initialize calculator with option parameters
        self.inputs = inputs
        self.d1 = None
        self.d2 = None
        self.calc_d_values()
    
    def calc_d_values(self):
        # Calculate d1 and d2 parameters for Black-Scholes formula
        S = self.inputs.spot_price
        K = self.inputs.strike_price
        T = self.inputs.time_to_expiry
        r = self.inputs.risk_free_rate
        sigma = self.inputs.volatility
        q = self.inputs.dividend_yield
        
        self.d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)
    
    def calc_price(self) -> float:
        # Calculate option price using Black-Scholes formula
        S = self.inputs.spot_price
        K = self.inputs.strike_price
        T = self.inputs.time_to_expiry
        r = self.inputs.risk_free_rate
        q = self.inputs.dividend_yield
        
        if self.inputs.option_type.lower() == 'call':
            price = (S * np.exp(-q * T) * norm.cdf(self.d1) - 
                    K * np.exp(-r * T) * norm.cdf(self.d2))
        else:  # put
            price = (K * np.exp(-r * T) * norm.cdf(-self.d2) - 
                    S * np.exp(-q * T) * norm.cdf(-self.d1))
        
        return float(price)
    
    def calc_delta(self) -> float:
        # Calculate Delta: ∂V/∂S
        q = self.inputs.dividend_yield
        T = self.inputs.time_to_expiry
        
        if self.inputs.option_type.lower() == 'call':
            delta = np.exp(-q * T) * norm.cdf(self.d1)
        else:
            delta = -np.exp(-q * T) * norm.cdf(-self.d1)
        
        return float(delta)
    
    def calc_gamma(self) -> float:
        # Calculate Gamma: ∂²V/∂S²
        S = self.inputs.spot_price
        T = self.inputs.time_to_expiry
        sigma = self.inputs.volatility
        q = self.inputs.dividend_yield
        
        gamma = (np.exp(-q * T) * norm.pdf(self.d1)) / (S * sigma * np.sqrt(T))
        return float(gamma)
    
    def calc_vega(self) -> float:
        # Calculate Vega: ∂V/∂σ
        S = self.inputs.spot_price
        T = self.inputs.time_to_expiry
        q = self.inputs.dividend_yield
        
        vega = S * np.exp(-q * T) * norm.pdf(self.d1) * np.sqrt(T)
        return float(vega / 100)  # Per 1% change in volatility
    
    def calc_theta(self) -> float:
        # Calculate Theta: ∂V/∂T
        S = self.inputs.spot_price
        K = self.inputs.strike_price
        T = self.inputs.time_to_expiry
        r = self.inputs.risk_free_rate
        sigma = self.inputs.volatility
        q = self.inputs.dividend_yield
        
        if self.inputs.option_type.lower() == 'call':
            theta = (-(S * norm.pdf(self.d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * norm.cdf(self.d2) +
                    q * S * np.exp(-q * T) * norm.cdf(self.d1))
        else:
            theta = (-(S * norm.pdf(self.d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * norm.cdf(-self.d2) -
                    q * S * np.exp(-q * T) * norm.cdf(-self.d1))
        
        return float(theta / 365)  # Per day
    
    def calc_greeks(self) -> Dict[str, float]:
        # Calculate option price and all Greeks in one call
        return {
            'price': self.calc_price(),
            'delta': self.calc_delta(),
            'gamma': self.calc_gamma(),
            'vega': self.calc_vega(),
            'theta': self.calc_theta()
        }
    
    def display_results(self):
        # Display result from calc_greeks()
        results = self.calc_greeks()
        
        print("=" * 60)
        print("BLACK-SCHOLES OPTION PRICING & GREEKS")
        print("=" * 60)
        print(f"\nOption Type: {self.inputs.option_type.upper()}")
        print(f"Spot Price: ${self.inputs.spot_price:.2f}")
        print(f"Strike Price: ${self.inputs.strike_price:.2f}")
        print(f"Time to Expiry: {self.inputs.time_to_expiry:.4f} years ({self.inputs.time_to_expiry*365:.0f} days)")
        print(f"Volatility: {self.inputs.volatility*100:.2f}%")
        print(f"Risk-Free Rate: {self.inputs.risk_free_rate*100:.2f}%")
        
        print(f"\n{'OPTION PRICE':.<40} ${results['price']:.4f}")
        print("\nGREEKS:")
        print(f"{'Delta (Δ) - Price sensitivity':.<40} {results['delta']:.6f}")
        print(f"{'Gamma (Γ) - Delta sensitivity':.<40} {results['gamma']:.6f}")
        print(f"{'Vega (ν) - Volatility sensitivity':.<40} {results['vega']:.6f}")
        print(f"{'Theta (Θ) - Time decay per day':.<40} {results['theta']:.6f}")
        print("=" * 60)


def main():
    # Example usage of the Black-Scholes calculator 
    # Define option parameters
    inputs = OptionInputs(
        spot_price=100.0,
        strike_price=105.0,
        time_to_expiry=0.5,  # 6 months
        volatility=0.25,      # 25% volatility
        risk_free_rate=0.05,  # 5% risk-free rate
        option_type='call',
        dividend_yield=0.0
    )
    
    # Create calculator and display results
    calculator = BlackScholesCalculator(inputs)
    calculator.display_results()
    
    # Get results as dictionary for further processing
    results = calculator.calc_greeks()
    
    print("\n\nExample: Calculating for a PUT option")
    print("-" * 60)
    
    put_inputs = OptionInputs(
        spot_price=100.0,
        strike_price=95.0,
        time_to_expiry=0.25,  # 3 months
        volatility=0.30,
        risk_free_rate=0.05,
        option_type='put'
    )
    
    put_calculator = BlackScholesCalculator(put_inputs)
    put_calculator.display_results()


if __name__ == "__main__":
    main()

