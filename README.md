# koeniglab-pmm-project
A repository storing research progress into parametric matrix models applied to finite volume physics.

# Parametric Matrix Model (PMM)

A research framework for learning parametric eigenvalue structures from sampled data.  
The Parametric Matrix Model (PMM) provides a way to approximate the spectral behavior of a physical system whose Hamiltonian depends smoothly on a control parameter (e.g. system size, coupling constant, or external field strength).

---

## Background and Motivation

In many physical systems, the eigenvalues of a Hamiltonian $H(\lambda)$ evolve continuously with respect to a tunable parameter $\lambda$.  
Computing spectra across a dense grid of parameter values can be computationally expensive. The **Parametric Matrix Model (PMM)** aims to learn an approximate functional form of $H(\lambda)$ from a small number of samples.

PMMs were originally motivated by another computational technique: **eigenvector continuation (EC)**. In EC, one faces the same problem: explicitly diagonalizing the Hamiltonian -- which is often massive -- for a dense grid of parameter 
values is too expensive. Instead, one diagonalizes the Hamiltonian at a few parameter values (sample points), and constructs a subspace from the eigenvectors at those points: $M=\text{span}(\lbrace v_i\rbrace_i)\subset\mathcal{H}$. 
Then, one projects the exact Hamiltonian at other parameter values onto this subspace to obtain a generalized eigenvalue problem: $H_{\text{proj}}v' = ESv'$ where $H_{\text{proj}}^_{ij}=\langle v_i|H|v_j\rangle$, $S_{ij}=\langle v_i | v_j\rangle$, and $v'$ is a coordinate vector
in the basis of the sample eigenvectors. This generalized eigenvalue problem has a much smaller dimension (its dimension is just the number of sample vectors taken), and is much easier to solve, than the parent eigenvalue problem.

This approach assumes that all of the eigenvectors at all parameter values of interest lie (approximately) in the span of the sample eigenvectors. This assumption is reasonable if one is interested only in the lowest few eigenstates: when $H$ is analytic in $\lambda$, 
so are its eigenvalues and eigenvectors. Expanding an eigenvector in a Taylor series about some reference point $\lambda_0$, 
```math
\psi'(\lambda) = \sum_n (\lambda-\lambda_0)^n \frac{\psi^{(n)}(\lambda_0)}{n!}
```
shows that eigenvectors at nearby parameter values differ only by higher-order derivatives of $\psi$. In finite difference terms, this means that $\psi(\lambda)$ can be well approximated as a linear combination of eigenvectors evaluated at a few nearby parameter points. 
In other words, the first few eigenvectors only explore a small, local subset of Hilbert space as you vary the parameter. Projecting on this subspace allows us to solve a lower dimensional problem than the parent problem

In PMMs, one does not explicitly project the parent Hamiltonian onto a reduced subspace. Instead, the "projected Hamiltonian" is learned directly. Given training data consisting of eigenpairs at sampled parameter values,
```math
H(\lambda_i) \psi_n^{(i)} = E_n^{(i)} \psi_n^{(i)},
```
the most basic PMM posits a parameterized form 
```math
H_\theta(\lambda) = A_\theta + \lambda B_\theta,
```
and learns $A_\theta, B_\theta$ by minimizing the difference between the model’s eigenvalues and the sampled true energies. 

When the underlying Hamiltonian depends affinely on $\lambda$, this is equivalent to training (rather than explicitly computing) the projected matrix used in EC. However, even if the parent Hamiltonian has a more complicated or unknown dependence on $\lambda$, this approach remains valid: it effectively learns a low-order Taylor expansion of the Hamiltonian in parameter space. Additional terms (e.g., $H_\theta(\lambda) = A + \lambda B + \lambda^2 C + ...)$ or alternate functional forms (e.g., $A + 1/\lambda B + ...)$ can be incorporated to capture more complex behavior.
Once trained, the PMM can predict spectra at parameter values it hasn’t seen before. Interpolation between known values has already been demonstrated, but here we investigate whether PMMs can also extrapolate beyond the training domain. While PMMs are far more general than this brief description suggests, the above summarizes their role in the present work.

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

