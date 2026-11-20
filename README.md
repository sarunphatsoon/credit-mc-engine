# Credit Loss Monte Carlo Engine (C++ / Python)

A multithreaded C++20 Monte Carlo engine for portfolio credit losses under the
Vasicek one-factor model (the model behind the Basel IRB capital formula),
exposed to Python via pybind11.

## What it does

- Simulates portfolio losses with a systematic factor plus obligor-specific shocks
- Handles heterogeneous portfolios (different PD, EAD and LGD per obligor)
- Reproducible: the same seed gives identical results for any thread count
- Validated against a pure NumPy implementation and the closed-form IRB formula

## Build

Requires a C++20 compiler, CMake 3.18+ and Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cmake -S . -B build -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
cmake --build build -j
cd python
python compare.py
python concentration.py
```

## Results

**Validation and speed** (1,000 obligors, 200,000 scenarios, Apple Silicon Mac):
the C++ engine matches the NumPy reference and the analytical expected loss,
and runs roughly 9-10x faster.

**Concentration effect** (1,000 obligors, 1M scenarios, PD 2%, LGD 45%, rho 0.15,
total exposure fixed). The IRB formula gives one VaR for every row because it
ignores concentration:

| Top-10 share of exposure | MC VaR (99.9%) | MC / IRB | ES (99.9%) |
|---|---|---|---|
| 1% (equal loans) | 79.65 | 1.004 | 93.96 |
| 10% | 80.18 | 1.011 | 95.10 |
| 20% | 83.09 | 1.047 | 98.49 |
| 30% | 88.55 | 1.116 | 104.23 |
| 40% | 95.18 | 1.200 | 112.00 |

IRB VaR is 79.35 in every row. With 40% of exposure in 10 loans, simulated
tail risk is about 20% above the IRB figure.

## Layout

- `include/creditmc/engine.hpp`, `src/engine.cpp`: simulation engine
- `src/bindings.cpp`: pybind11 module `creditmc_core`
- `python/reference.py`: NumPy reference implementation
- `python/compare.py`: validation and benchmark
- `python/concentration.py`: concentration experiment