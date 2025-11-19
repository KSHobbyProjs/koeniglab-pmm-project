# koeniglab-pmm-project
A repository storing a program that runs a parametric matrix model on a physical model and plots the results.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/KSHobbyProjs/koeniglab-pmm-project.git
cd koeniglab-pmm-project
pip install -r requirements.txt
```
Dependencies include `numpy`, `scipy`, `matplotlib`, `jax`

---

## Usage

Run an experiment using `main.py`.  
This script samples exact data, trains a PMM, and plots predictions.

```bash
./main.py \
    --model_name gaussian.Gaussian1d \
    --epochs 5000 \
    --sample_Ls 5.0,20.0:50 \
    --predict_Ls None \
    --k_num_sample 3 \
    --k_num_predict 3 \
    --store_loss 100
```
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

## Utilities

1. Load all energy data from all physics models by running `python -m scripts.get_exact_data` from root.
2. Load all EC predictions by running `python -m scripts.get_ec_predicted_data`.

---

## Extending the Code

To add a new model:
1. Create a new file under `src/physics_models/` (e.g., `double_well.py`).
2. Create a class within this file that subclasses `BaseModel`.
3. Change
   ```python
   def construct_H(self, L): ...
   ```
   so that it constructs the Hamiltonian for your model as it depends on the parameter $L$.
4. If you want to pre-load exact eigenvalue data for your model, add a line to `src/scripts/get_exact_data.py`.

To add a new PMM variant:
1. Add a new class under `src/algorithms/pmm.py` that subclasses `PMM`.
2. Modify
   ```python
   def loss(params, Ls, energies, l2): ...
   def get_basis(Ls, num_primary): ...
   ```
   to change how the loss is computed (default is mean squared error between predicted eigenvalues and sample eigenvalues,
   and how the basis is constructed (default is affine $H_\theta=A_\theta+\lambda B_\theta$).
---

