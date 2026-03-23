
import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict


#dataclass
class OptionInputs:
    #Structure for passing initial option parameters.
    spot_price: float
    strike_price: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float
    option_type: str
    dividend_yield: float = 0.0


class OptionEngine:
    #Advanced Quantitative Engine for Black-Scholes-Merton Pricing, 
    # Greeks, Implied Volatility, and Scenario-Based Risk Analysis.
    
    def __init__(self, inputs: OptionInputs):
        self.inputs = inputs

    def _calc_d_values(self, S, sigma):
        #Vectorized calculation of d1 and d2.
        K, T, r, q = self.inputs.strike_price, self.inputs.time_to_expiry, self.inputs.risk_free_rate, self.inputs.dividend_yield
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def calc_price(self, S=None, sigma=None):
        #Calculates option price using BSM formula (Vectorized).
        S = S if S is not None else self.inputs.spot_price
        sigma = sigma if sigma is not None else self.inputs.volatility
        K, T, r, q = self.inputs.strike_price, self.inputs.time_to_expiry, self.inputs.risk_free_rate, self.inputs.dividend_yield
        
        d1, d2 = self._calc_d_values(S, sigma)
        
        if self.inputs.option_type.lower() == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def calc_greeks(self, S=None, sigma=None) -> Dict[str, float]:
        #Returns the full suite of analytical Greeks.
        S = S if S is not None else self.inputs.spot_price
        sigma = sigma if sigma is not None else self.inputs.volatility
        K, T, r, q = self.inputs.strike_price, self.inputs.time_to_expiry, self.inputs.risk_free_rate, self.inputs.dividend_yield
        
        d1, d2 = self._calc_d_values(S, sigma)
        
        # Delta
        if self.inputs.option_type.lower() == 'call':
            delta = np.exp(-q * T) * norm.cdf(d1)
        else:
            delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
            
        # Gamma and Vega
        gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        return {
            "price": self.calc_price(S, sigma),
            "delta": delta,
            "gamma": gamma,
            "vega": vega / 100  # Per 1% change
        }

    def get_implied_vol(self, market_price, precision=1e-6):
        #Newton-Raphson solver for Implied Volatility.
        S = self.inputs.spot_price
        sigma = 0.3  # Initial guess
        for _ in range(100):
            p = self.calc_price(S, sigma)
            v = (self.calc_greeks(S, sigma)['vega'] * 100) # Convert back from %
            diff = market_price - p
            if abs(diff) < precision:
                return sigma
            if v == 0: break
            sigma += diff / v
        return sigma

    def run_scenario_analysis(self, price_range=0.1, vol_range=0.05):
        #Generates a 2D Profit/Loss Risk Matrix.
        spot, vol = self.inputs.spot_price, self.inputs.volatility
        
        S_range = np.linspace(spot * (1 - price_range), spot * (1 + price_range), 11)
        V_range = np.linspace(max(0.01, vol - vol_range), vol + vol_range, 11)
        S_mesh, V_mesh = np.meshgrid(S_range, V_range)

        # Vectorized price calculation over the entire mesh
        scenario_prices = self.calc_price(S_mesh, V_mesh)
        current_val = self.calc_price(spot, vol)
        pnl_matrix = scenario_prices - current_val

        # Formatting the Risk Matrix
        df = pd.DataFrame(
            pnl_matrix, 
            index=[f"Vol {v*100:.1f}%" for v in V_range],
            columns=[f"Spot ${s:.2f}" for s in S_range]
        )
        return df

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Inputs
    my_inputs = OptionInputs(
        spot_price=100.0,
        strike_price=105.0,
        time_to_expiry=0.5,
        volatility=0.25,
        risk_free_rate=0.05,
        option_type='call'
    )

    # 2. Initialize Engine
    engine = OptionEngine(my_inputs)
    
    # 3. Display Core Results
    greeks = engine.calc_greeks()
    print(f"Option Price: ${greeks['price']:.4f} | Delta: {greeks['delta']:.4f}")
    
    # 4. Run and Display Scenario Analysis
    print("\nSCENARIO ANALYSIS: P&L RISK MATRIX")
    print("=" * 65)
    risk_report = engine.run_scenario_analysis()
    print(risk_report.round(2))
