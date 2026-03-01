# **`README.md`**

# Constrained Portfolio Optimization via Quantum Approximate Optimization Algorithm (QAOA) with XY-Mixers and Trotterized Initialization

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2602.14827-b31b1b.svg)](https://arxiv.org/abs/2602.14827)
[![Journal](https://img.shields.io/badge/Journal-ArXiv%20Preprint-003366)](https://arxiv.org/abs/2602.14827)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm)
[![Discipline](https://img.shields.io/badge/Discipline-Financial%20Engineering%20%7C%20Quantum%20Computing-00529B)](https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm)
[![Data Sources](https://img.shields.io/badge/Data-Yahoo%20Finance%20%7C%20Bloomberg-lightgrey)](https://finance.yahoo.com/)
[![Core Method](https://img.shields.io/badge/Method-QAOA%20%7C%20Combinatorial%20Optimization-orange)](https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm)
[![Analysis](https://img.shields.io/badge/Analysis-Walk--Forward%20Backtesting-red)](https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm)
[![Validation](https://img.shields.io/badge/Validation-Out--of--Sample%20Performance-green)](https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm)
[![Robustness](https://img.shields.io/badge/Robustness-Trotterized%20Initialization-yellow)](https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-%23000000.svg?style=flat&logo=Xanadu&logoColor=white)](https://pennylane.ai/)
[![D-Wave](https://img.shields.io/badge/D--Wave%20Ocean-%2300AEEF.svg?style=flat&logo=D-Wave&logoColor=white)](https://docs.ocean.dwavesys.com/)
[![YAML](https://img.shields.io/badge/YAML-%23CB171E.svg?style=flat&logo=yaml&logoColor=white)](https://yaml.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=flat&logo=Jupyter&logoColor=white)](https://jupyter.org/)
[![Open Source](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-brightgreen)](https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm)

**Repository:** `https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional Python codebase. It implements the research from the 2026 paper titled **"Constrained Portfolio Optimization via Quantum Approximate Optimization Algorithm (QAOA) with XY-Mixers and Trotterized Initialization: A Hybrid Approach for Direct Indexing"** by:

*   **Javier Mancilla** (SquareOne Capital)
*   **Theodoros D. Bouloumis** (Aristotle University of Thessaloniki)
*   **Frederic Goguikian** (SquareOne Capital)

This project gives you a complete framework to reproduce the paper's findings. It provides a clear, step-by-step pipeline. The code handles everything from cleaning raw market data to running quantum simulations. Finally, it tests the quantum results against classical methods to prove their value.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `execute_quantum_hybrid_portfolio_optimization`](#key-callable-execute_quantum_hybrid_portfolio_optimization)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides Python code for the framework presented in Mancilla et al. (2026). The main file is the Jupyter Notebook `constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm_draft.ipynb`. It contains over 27 functions to reproduce the paper's results.

The code solves a major problem in finance: Cardinality Constrained Portfolio Optimization for "Direct Indexing." When you must pick exactly $K$ assets from a pool of $N$, standard Markowitz math fails. The problem becomes NP-hard.

The paper offers a Hard-Constraint QAOA solution. Standard QAOA uses soft penalties that warp the energy landscape. Instead, this new method enforces strict rules using Dicke states and XY-mixers. Our code brings this theory to life. It does the following:
-   **Checks** data to ensure it is clean and free of look-ahead bias.
-   **Builds** the quantum state using Dicke initialization to keep the math valid.
-   **Runs** the quantum circuit using PennyLane. It uses a special start-up schedule to avoid flat gradients (Barren Plateaus).
-   **Compares** the quantum solver against Simulated Annealing (SA) and Hierarchical Risk Parity (HRP).
-   **Scores** the performance using a strict backtest. It calculates Sharpe Ratios, Drawdowns, and Turnover after trading fees.

## Theoretical Background

The methods combine Financial Econometrics, Quantum Computing, and Convex Optimization.

**1. The Main Goal:**
We want to select exactly $K$ assets to balance risk and return:
$$ \min_{x \in \{0,1\}^N} \left( q x^\top \Sigma x - (1-q) \mu^\top x \right) \quad \text{s.t.} \quad \sum_{i=1}^N x_i = K $$

**2. The Quantum Setup:**
*   **Dicke State Start:** The system begins in an equal mix of all valid portfolios:
    $$ |\psi_0\rangle = |D^K_N\rangle = \binom{N}{K}^{-1/2} \sum_{|x|=K} |x\rangle $$
*   **XY-Mixer:** The operator swaps assets without changing the total number selected:
    $$ H_{XY} = \sum_{(i,j) \in E} (X_i X_j + Y_i Y_j) $$

**3. Smart Initialization:**
To keep the math learning smoothly, we start the parameters on a linear ramp:
$$ \gamma_l = \frac{l}{p}\Delta t, \quad \beta_l = \left(1 - \frac{l}{p}\right)\Delta t $$

Below is a diagram that summarizes the approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm/blob/main/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm_ipo_main.png" alt="QAOA-XY System Architecture" width="100%">
</div>

## Features

The Jupyter Notebook runs the full research pipeline. Key features include:

-   **Modular Design:** The code breaks down into 27 clear, math-focused tasks.
-   **Easy Setup:** You control all settings (like risk limits and costs) from one `config.yaml` file.
-   **Strict Data Checks:** The system tests data for errors, timezones, and look-ahead bias before running.
-   **Fast Quantum Simulation:** We use PennyLane to simulate quantum states. A custom Adam loop speeds up training.
-   **Strong Baselines:** The code compares quantum results against Simulated Annealing (using D-Wave tools) and Hierarchical Risk Parity.
-   **Final Proofs:** The run ends with strict math checks. It proves every portfolio followed the rules.

## Methodology Implemented

The code follows the exact steps from the paper:

1.  **Data Prep (Tasks 1-4):** Checks settings, cleans prices, and builds a strict trading calendar.
2.  **Time & Returns (Tasks 5-6):** Slices data into safe windows and finds daily log returns.
3.  **Math Estimates (Tasks 7-9):** Finds expected returns ($\mu$) and risk matrices ($\Sigma$). It also tracks past holdings.
4.  **Classical Baseline (Tasks 10-12):** Builds a QUBO matrix and runs Simulated Annealing to find a baseline portfolio.
5.  **Quantum Setup (Tasks 13-15):** Prepares the Dicke state, builds the XY-mixer, and maps finance math to quantum physics.
6.  **Quantum Training (Tasks 16-18):** Trains the quantum circuit at different depths. It filters the results and picks the best portfolio.
7.  **Weight Allocation (Tasks 19-21):** Assigns exact cash weights to the chosen assets to maximize the Sharpe Ratio.
8.  **Performance Tracking (Tasks 22-24):** Calculates returns, tracks turnover, deducts trading fees, and scores the strategy.
9.  **Final Checks (Tasks 25-27):** Saves all files to disk and runs final math proofs to ensure accuracy.

## Core Components (Notebook Structure)

The project lives inside a single Jupyter Notebook: `constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm_draft.ipynb`. The notebook is built as a logical pipeline. It has separate functions for each of the 27 major tasks. Every function is self-contained, fully documented, and ready for professional use.

## Key Callable: `execute_quantum_hybrid_portfolio_optimization`

The project revolves around one main function:

-   **`execute_quantum_hybrid_portfolio_optimization`:** This master function runs the entire pipeline from start to finish. One call to this function handles data cleaning, math estimates, quantum simulation, classical weighting, and final checks.

## Prerequisites

-   Python 3.9+
-   Core dependencies: `pandas`, `numpy`, `scipy`, `pennylane`, `dimod`, `dwave-neal`, `PyPortfolioOpt`, `pyyaml`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm.git
    cd constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install pandas numpy scipy pennylane dimod dwave-neal PyPortfolioOpt pyyaml
    ```

## Input Data Structure

The pipeline needs one main DataFrame (`raw_price_df`):

-   **Index:** `DatetimeIndex` (trading days moving forward in time).
-   **Columns:** Exactly 10 string names matching your setup (e.g., `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `JPM`, `V`, `TSLA`, `UNH`, `LLY`, `XOM`).
-   **Values:** `float64` numbers showing adjusted closing prices. They must be greater than zero.

*Note: The usage example below includes a tool to generate fake market data. You can use this to test the code if you do not have live Yahoo Finance data.*

## Usage

The code below shows how to create fake market data, load your settings, and run the main pipeline.

```python
import pandas as pd
import numpy as np
import yaml
import os

# Assuming all pipeline callables are loaded in the current namespace

# 1. Generate Synthetic Price Data (Geometric Brownian Motion)
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", end="2025-12-31", freq='B')
universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "V", "TSLA", "UNH", "LLY", "XOM"]
prices = np.zeros((len(dates), len(universe)))
prices[0] = np.random.uniform(100, 200, size=len(universe))
dt = 1.0 / 252.0
for t in range(1, len(dates)):
    Z = np.random.standard_normal(len(universe))
    prices[t] = prices[t-1] * np.exp((0.10 - 0.5 * 0.20**2) * dt + 0.20 * np.sqrt(dt) * Z)

raw_price_df = pd.DataFrame(prices, index=dates, columns=universe).astype(np.float64)

# 2. Load Configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# 3. Execute the Pipeline
output_directory = "./quantum_portfolio_artifacts"
study_artifacts = execute_quantum_hybrid_portfolio_optimization(
    raw_price_df=raw_price_df,
    raw_config=config,
    output_dir=output_directory
)

# 4. Inspect Results
print("\n[Fidelity Audit]")
print(f"Mathematical Constraints Verified: {study_artifacts['fidelity_verified']}")

print("\n[Financial Performance Metrics (2025)]")
metrics_df = pd.DataFrame.from_dict(study_artifacts["results_bundle"]["financial_metrics"], orient="index")
print(metrics_df)
```

## Output Structure

The pipeline returns a large dictionary containing:
-   **`validated_config`**: The checked and approved settings.
-   **`data_validation_report`**: Notes from the data cleaning phase.
-   **`cleaned_price_df`**: The final price matrix used for the study.
-   **`rebalance_dates`**: The locked trading schedule.
-   **`results_bundle`**: A nested dictionary containing:
    -   `financial_metrics`: Total Return, Volatility, Sharpe, MDD, and Turnover for QAOA, SA, and HRP.
    -   `depth_scaling_table`: Quantum training data (Cost, Iterations, Gradients) across depths $p=1 \dots 6$.
    -   `persisted_files`: A list of all files saved to your hard drive (Parquet, NPY, JSON).
-   **`fidelity_verified`**: A true/false flag confirming all math rules were followed.

## Project Structure

```
constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm/
│
├── constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm_draft.ipynb   # Main implementation notebook
├── config.yaml                                                                                     # Master configuration file
├── requirements.txt                                                                                # Python package dependencies
│
├── LICENSE                                                                                         # MIT Project License File
└── README.md                                                                                       # This file
```

## Customization

You can easily change the pipeline using the `config.yaml` file. Users can modify settings such as:
-   **Financial Setup:** Asset list, lookback window ($L$), and risk aversion ($q$).
-   **Quantum Architecture:** Circuit depths ($p$), start-up bounds, and Adam optimizer step sizes.
-   **Classical Solvers:** SA reads/sweeps and SLSQP weight limits.
-   **Friction Models:** Trading cost basis points ($\tau$) and continuity bonus ($\kappa$).

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. You must follow PEP 8 rules, use strict type hinting, and write clear docstrings.

## Recommended Extensions

Future updates could include:
-   **Hardware Execution:** Moving the `default.qubit` simulator to real, noisy quantum hardware (like IBM or IonQ) using Amazon Braket or Azure Quantum.
-   **Multi-Period Rules:** Adding turnover penalties directly into the quantum math to optimize net returns natively on the QPU.
-   **Alternative Mixers:** Testing ring-graph XY-mixers to reduce circuit depth and gate counts for near-term hardware.
-   **Expanded Universes:** Scaling the simulation beyond $N=10$ using tensor network simulators (like `default.tensor`).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original paper:

```bibtex
@article{mancilla2026constrained,
  title={Constrained Portfolio Optimization via Quantum Approximate Optimization Algorithm (QAOA) with XY-Mixers and Trotterized Initialization: A Hybrid Approach for Direct Indexing},
  author={Mancilla, Javier and Bouloumis, Theodoros D. and Goguikian, Frederic},
  journal={arXiv preprint arXiv:2602.14827},
  year={2026}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). Constrained Portfolio Optimization via QAOA with XY-Mixers: An Open Source Implementation.
GitHub repository: https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm
```

## Acknowledgments

-   Credit to **Javier Mancilla, Theodoros D. Bouloumis, and Frederic Goguikian** for the foundational research that forms the entire basis for this computational replication.
-   This project is built upon the exceptional tools provided by the open-source quantum and scientific Python communities. Sincere thanks to the developers of **PennyLane, D-Wave Ocean, PyPortfolioOpt, Pandas, NumPy, and SciPy**.