# Black-Scholes-Merton Risk Engine

A high-performance quantitative finance library for **real-time sensitivity analysis** and **scenario-based risk management**. This project implements the closed-form Black-Scholes-Merton (BSM) model to value European options and estimate their associated Greeks using analytical derivatives.



## Key Features

* **Full Greeks Suite:** Analytical calculation of **Delta ($\Delta$)**, **Gamma ($\Gamma$)**, **Vega ($\nu$)**, and **Theta ($\Theta$)** for European Call and Put options.
* **Implied Volatility (IV) Estimation:** High-speed numerical root-finding using the **Newton-Raphson algorithm** (Convergence tolerance: $10^{-6}$).
* **Scenario-Based Risk Analysis:** Bivariate sensitivity engine that generates Profit/Loss (P&L) matrices across varying spot price and volatility scenarios.
* **High-Performance Vectorization:** Built on `NumPy` to handle large-scale "meshgrids" for real-time risk profiling.



## Technical Stack

* **Language:** Python 3.9+
* **Core Math:** `NumPy` (Vectorized Array Processing), `SciPy` (Normal Distributions)
* **Data Analysis:** `Pandas` (Risk Reporting & Matrix Formatting)
* **Visualization:** `Seaborn` & `Matplotlib` (Heatmap Generation)

## Core Function

### 1. The Pricing Engine
The engine uses a stateless, vectorized approach to calculate option premiums, accounting for risk-free rates ($r$) and continuous dividend yields ($q$).

### 2. Implied Volatility Solver
Since $\sigma$ is unobservable in the market, the engine "backs out" the market's expected volatility by iteratively solving the BSM formula using the Newton-Raphson method:
$$\sigma_{n+1} = \sigma_n - \frac{C(\sigma_n) - C_{market}}{\text{Vega}(\sigma_n)}$$



### 3. Scenario Analysis & Stress Testing
The engine generates a 2D "Risk Mesh" to visualize how an option's value decays or expands under different market stress tests. This is a crucial tool for **Delta-hedging** and managing **Vega-exposure**.
