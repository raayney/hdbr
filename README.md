# basket-dispersion-ml
# High-Dimensional Basket Dispersion Trading via Model-Free Deep Learning

This project builds a model-free pricing and trading framework for
variance dispersion trades on the EURO STOXX 50 index and its
constituents, using martingale optimal transport (MOT) and deep neural
networks.

## Project status

Phase 1 (design and documentation) in progress.

Key docs:

- `docs/01_problem_and_payoff.md` – dispersion payoff and trading signal
- `docs/02_model_free_and_nn_formulation.md` – model-free bounds and NN surrogate
- `docs/03_data_specification.md` – data sources, Yahoo underlyings and synthetic options

## Planned structure

- `docs/` – all design and research documentation
- `src/` – data, pricing, models, and backtesting code
- `data/` – local data files (not committed to git)
- `notebooks/` – exploratory analysis and experiments
- `config/` – configuration files (paths, hyperparameters)

Implementation will follow the design documents in Phase 2.
