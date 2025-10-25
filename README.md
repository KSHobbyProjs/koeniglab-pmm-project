# koeniglab-pmm-project
A repository storing research progress into parametric matrix models applied to finite volume physics.

# Parametric Matrix Model (PMM)

A research framework for learning parametric eigenvalue structures from sampled data.  
The Parametric Matrix Model (PMM) provides a way to approximate the spectral behavior of a physical system whose Hamiltonian depends smoothly on a control parameter (e.g. system size, coupling constant, or external field strength).

---

## Background and Motivation

In many physical systems, the eigenvalues of a Hamiltonian $H(\lambda)$ evolve continuously with respect to a tunable parameter $\lambda$.  
Computing spectra across a dense grid of parameter values can be computationally expensive. The **Parametric Matrix Model (PMM)** aims to learn an approximate functional form of $H(\lambda)$ from a small number of samples.

PMMs were originally motivated by another computational technique: eigenvector continuation (EC). In EC, one faces the same problem: explicitly diagonalizing the Hamiltonian, which is often massive, for a dense grid of parameter 
values is often too expensive. Instead, one can diagonalize the Hamiltonian at a few parameter values (sample points), and construct a subspace from the eigenvectors at those parameter values, $M_\text{span}$. 
Then, one can project the Hamiltonian onto this subspace to get a generalized eigenvalue problem $H_{\text{proj}}v = ESv$

Given training data consisting of eigenpairs at sampled parameter values,
$$
H(\lambda_i) \psi_n^{(i)} = E_n^{(i)} \psi_n^{(i)},
$$
the PMM assumes a parameterized matrix form
$$
H_\theta(\lambda) = A_\theta + \lambda B_\theta,
$$
and learns \( A_\theta, B_\theta \) by minimizing the difference between the model’s eigenvalues and the sampled true energies.  
Once trained, the PMM can predict spectra at unseen parameter values.

---

## Project Structure

```
koeniglab-pmm-project/
├── src/
│   ├── algorithms/
│   │   ├── pmm.py                    # Core PMM algorithm (matrix construction, training, prediction)
│   │   └── ec.py                     # Core EC algorithm (sampling, projection, prediction)
│   ├── physics_models/
│   │   ├── base_model.py             # Example physical model: 1D Gaussian potential
│   │   ├── gaussian.py 
│   │   ├── ising.py
│   │   └── noninteracting_spins.py
│   ├── processing/
│   │   ├── process_pmm.py            # Orchestrates training, prediction, saving results
│   │   ├── process_exact.py          # Computes exact eigenpairs for reference
│   │   └── process_ec.py             # Orchestrates prediction and comparison through EC algorithm
│   └── utils/
│       ├── paths.py                  # Directory and filename utilities
│       ├── plot.py                   # Plot helpers for spectra and losses
│       ├── math.py                   # Data normalization and preprocessing
│       ├── io.py
│       └── misc.py
│
├── main.py                           # Entry point for running experiments
│
├── data/
│   └── experiments/                  # Each experiment gets its own timestamped subdirectory
│
├── results/
│
├── notebooks/                        # (Optional) exploratory analysis, visualization
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KSHobbyProjs/koeniglab-pmm-project.git
cd koeniglab-pmm-project
pip install -r requirements.txt
```

Typical dependencies include:
- `numpy`
- `scipy`
- `matplotlib`
- `jax`

---

## Usage

Run an experiment using the provided `main.py` script.  
This script samples exact data, trains a PMM, and plots predictions.

```bash
python -m src.main \
    --model_name gaussian \
    --epochs 5000 \
    --sample_Ls 1.0 1.5 2.0 \
    --predict_Ls 2.5 3.0 \
    --k_num_sample 3 \
    --k_num_predict 3 \
    --store_loss 100
```

### Example Arguments
- `--model_name`: Which exact model to use (`gaussian`, etc.)
- `--sample_Ls`: Parameter values at which exact data is computed
- `--predict_Ls`: Parameter values where predictions are evaluated
- `--epochs`: Number of PMM training iterations
- `--store_loss`: How often to record loss during training
- `--k_num_sample`, `--k_num_predict`: Number of eigenvalues to train and predict

All results are automatically stored under:
```
data/experiments/<model_name>/<timestamp>/
```

---

## Example Workflow

1. Compute or load exact spectra using `process_exact`.
2. Train a PMM using a small subset of sampled parameter values.
3. Predict eigenvalues at unseen parameters.
4. Plot results to compare PMM predictions vs. exact data.

Example plotting utilities:
```python
from src.utils.plotting import plot_spectra, plot_loss_curve
```

---

## Extending the Framework

To add a new model:
1. Create a new file under `src/models/` (e.g., `double_well.py`).
2. Implement functions:
   ```python
   def sample(L, dim): ...
   def exact_eigenpairs(L, dim): ...
   ```
3. Register your model name in `process_exact`.

To add a new PMM variant:
1. Add a new class under `src/algorithms/pmm/`.
2. Modify `process_pmm.py` to reference your new model by name.

---

