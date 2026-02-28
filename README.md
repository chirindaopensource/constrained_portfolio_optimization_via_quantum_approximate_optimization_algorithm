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

This repository contains an **independent**, professional-grade Python implementation of the research methodology from the 2026 paper entitled **"Constrained Portfolio Optimization via Quantum Approximate Optimization Algorithm (QAOA) with XY-Mixers and Trotterized Initialization: A Hybrid Approach for Direct Indexing"** by:

*   **Javier Mancilla** (SquareOne Capital)
*   **Theodoros D. Bouloumis** (Aristotle University of Thessaloniki)
*   **Frederic Goguikian** (SquareOne Capital)

The project provides a complete, end-to-end computational framework for replicating the paper's findings. It delivers a modular, auditable, and extensible pipeline that executes the entire hybrid quantum-classical workflow: from the ingestion and rigorous validation of financial market data to the formulation and simulation of constraint-preserving quantum circuits, culminating in comprehensive out-of-sample evaluation against classical heuristics.

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

This project provides a Python implementation of the analytical framework presented in Mancilla et al. (2026). The core of this repository is the iPython Notebook `constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm_draft.ipynb`, which contains a comprehensive suite of 27+ functions to replicate the paper's findings. 

The pipeline addresses the critical challenge of **Cardinality Constrained Portfolio Optimization** in the context of "Direct Indexing." Selecting exactly $K$ assets from a universe of $N$ transforms standard convex Markowitz optimization into an NP-hard combinatorial problem. 

The paper proposes a **Hard-Constraint QAOA** formulation. Unlike standard QAOA implementations that rely on soft penalty terms (which distort the energy landscape), this approach enforces constraints strictly via the quantum ansatz itself using Dicke states and XY-mixers. This codebase operationalizes the proposed solution:
-   **Validates** data integrity using strict schema checks and temporal causality enforcement.
-   **Engineers** the quantum state using Dicke state initialization to confine evolution to the feasible subspace.
-   **Simulates** the quantum circuit using PennyLane, employing a Trotterized parameter initialization to mitigate Barren Plateaus.
-   **Benchmarks** the quantum solver against Simulated Annealing (SA) and Hierarchical Risk Parity (HRP).
-   **Evaluates** performance via rigorous out-of-sample walk-forward backtesting, computing Sharpe Ratios, Drawdowns, and Turnover net of transaction costs.

## Theoretical Background

The implemented methods combine techniques from Financial Econometrics, Quantum Information Science, and Convex Optimization.

**1. The Combinatorial Objective:**
The objective is to select exactly $K$ assets to minimize the risk-return trade-off:
$$ \min_{x \in \{0,1\}^N} \left( q x^\top \Sigma x - (1-q) \mu^\top x \right) \quad \text{s.t.} \quad \sum_{i=1}^N x_i = K $$

**2. Constraint-Preserving Quantum Ansatz:**
*   **Dicke State Initialization:** The system begins in an equal superposition of all valid portfolios:
    $$ |\psi_0\rangle = |D^K_N\rangle = \binom{N}{K}^{-1/2} \sum_{|x|=K} |x\rangle $$
*   **XY-Mixer Hamiltonian:** The evolution operator performs partial SWAPs, commuting with the number operator to preserve the Hamming weight:
    $$ H_{XY} = \sum_{(i,j) \in E} (X_i X_j + Y_i Y_j) $$

**3. Trotterized Initialization:**
To avoid vanishing gradients in deep circuits, parameters are initialized via an adiabatic linear ramp:
$$ \gamma_l = \frac{l}{p}\Delta t, \quad \beta_l = \left(1 - \frac{l}{p}\right)\Delta t $$

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm/blob/main/constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm_ipo_main.png" alt="QAOA-XY System Architecture" width="100%">
</div>

## Features

The provided iPython Notebook implements the full research pipeline, including:

-   **Modular, 27-Task Architecture:** The pipeline is decomposed into highly specialized, mathematically rigorous functions.
-   **Configuration-Driven Design:** All study parameters (risk aversion, circuit depths, transaction costs) are managed in an external `config.yaml` file.
-   **Rigorous Data Validation:** A multi-stage validation process checks schema integrity, timezone consistency, and strict causality (zero look-ahead bias).
-   **Advanced Quantum Simulation:** Uses `PennyLane` for statevector simulation, featuring a highly optimized single-pass `value_and_grad` Adam training loop.
-   **Classical Baselines:** Integrates `dimod` and `neal` for Simulated Annealing via QUBO, and `PyPortfolioOpt` for Hierarchical Risk Parity.
-   **Institutional Fidelity Assertions:** The pipeline concludes with mathematical proofs asserting that all generated portfolios strictly obeyed the $K$-hot and sum-to-one constraints.

## Methodology Implemented

The core analytical steps directly implement the methodology from the paper:

1.  **Data Engineering (Tasks 1-4):** Validates the configuration, cleanses the raw price matrix (ffill/dropna), and freezes a deterministic, causal rebalance calendar.
2.  **Temporal & Return Infrastructure (Tasks 5-6):** Extracts strict $[t-L, t)$ causal windows and computes daily logarithmic returns.
3.  **Econometric Estimation (Tasks 7-9):** Computes annualized expected returns ($\mu$) and Ledoit-Wolf shrinkage covariance matrices ($\Sigma$), and manages the temporal continuity state.
4.  **Simulated Annealing Baseline (Tasks 10-12):** Constructs the penalized QUBO matrix, executes the `neal` sampler, and filters for feasible candidates.
5.  **QAOA Ansatz Construction (Tasks 13-15):** Prepares the Dicke statevector, builds the complete-graph XY-mixer, and maps the financial moments to the Ising Cost Hamiltonian.
6.  **QAOA Training & Selection (Tasks 16-18):** Executes the Trotterized Adam optimization loop across depths $p \in \{1 \dots 6\}$, extracts exact statevector probabilities, filters via bitwise operations, and selects the global optimum.
7.  **Continuous Allocation (Tasks 19-21):** Solves the Sharpe-max problem via SLSQP on the selected subsets, with a robust fallback to HRP.
8.  **Performance Accounting (Tasks 22-24):** Computes holding-period returns, $L_1$ turnover, applies 5 bps transaction costs, and aggregates final financial metrics and depth-scaling diagnostics.
9.  **Orchestration & Verification (Tasks 25-27):** Serializes all artifacts to disk and executes the final mathematical fidelity assertions.

## Core Components (Notebook Structure)

The project is contained within a single, comprehensive Jupyter Notebook: `constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm_draft.ipynb`. The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 27 major tasks. All functions are self-contained, fully documented with type hints and docstrings, and designed for professional-grade execution.

## Key Callable: `execute_quantum_hybrid_portfolio_optimization`

The project is designed around a single, top-level user-facing interface function:

-   **`execute_quantum_hybrid_portfolio_optimization`:** This master orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire computational portion of the project, managing data flow between validation, econometric estimation, quantum simulation, classical allocation, and final fidelity verification.

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

The pipeline requires a single primary DataFrame (`raw_price_df`):

-   **Index:** `DatetimeIndex` (monotonically increasing trading days).
-   **Columns:** Exactly 10 string identifiers matching the configured universe (e.g., `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `JPM`, `V`, `TSLA`, `UNH`, `LLY`, `XOM`).
-   **Values:** `float64` representing auto-adjusted closing prices (strictly positive).

*Note: The usage example below includes a synthetic data generator using Geometric Brownian Motion for testing purposes if access to live Yahoo Finance data is unavailable.*

## Usage

The following snippet demonstrates how to generate synthetic data, load the configuration, and execute the top-level orchestrator.

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

The pipeline returns a comprehensive dictionary containing:
-   **`validated_config`**: The strictly typed configuration object.
-   **`data_validation_report`**: Telemetry from the data cleansing phase.
-   **`cleaned_price_df`**: The canonical price matrix used for the study.
-   **`rebalance_dates`**: The frozen temporal schedule.
-   **`results_bundle`**: A nested dictionary containing:
    -   `financial_metrics`: Total Return, Volatility, Sharpe, MDD, Turnover for QAOA, SA, and HRP.
    -   `depth_scaling_table`: Aggregated quantum telemetry (Cost, Iterations, Gradient Norms) across depths $p=1 \dots 6$.
    -   `persisted_files`: A list of all artifacts serialized to disk (Parquet, NPY, JSON).
-   **`fidelity_verified`**: A boolean confirming all mathematical constraints were upheld.

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

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **Financial Setup:** Asset universe, lookback window ($L$), and risk aversion ($q$).
-   **Quantum Architecture:** Circuit depths ($p$), Trotterization bounds, and Adam optimizer step sizes.
-   **Classical Solvers:** SA reads/sweeps and SLSQP allocation bounds.
-   **Friction Models:** Transaction cost basis points ($\tau$) and continuity bonus ($\kappa$).

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and comprehensive docstrings is required.

## Recommended Extensions

Future extensions could include:
-   **Hardware Execution:** Migrating the `default.qubit` statevector simulator to noisy quantum hardware (e.g., IBM, IonQ) via Amazon Braket or Azure Quantum.
-   **Multi-Period Regularization:** Incorporating explicit turnover penalties directly into the Ising Hamiltonian to optimize the net-of-fee objective natively on the QPU.
-   **Alternative Mixers:** Exploring ring-graph XY-mixers to reduce circuit depth and CNOT gate counts for near-term hardware compatibility.
-   **Expanded Universes:** Scaling the simulation beyond $N=10$ using tensor network simulators (e.g., `default.tensor`).

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

--

*This README was generated based on the structure and content of the `constrained_portfolio_optimization_via_quantum_approximate_optimization_algorithm_draft.ipynb` notebook and follows best practices for research software documentation.*
