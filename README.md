<div align="center">

# utorch

<img src="docs/assets/logo.png" width="200" alt="utorch logo" />

**A Minimalist Deep Learning Engine & Autograd Implemented from Scratch**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-8A2BE2.svg)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


---

> **Current Branch: `feat/04-tensor-autograd`**
> This branch scales the autograd engine from 0-D scalars to multi-dimensional tensors. It introduces Vector-Jacobian Products (VJPs), automated reverse broadcasting (`unbroadcast`), dimensional reductions (`sum`, `mean`), matrix multiplication (`@`).

---

## Lecture Overview & Branch Progression

`utorch` is an educational implementation of automatic differentiation and deep learning primitives designed for teaching PyTorch internals from first principles.

To make the learning journey manageable, the project is structured across a sequence of focused Git branches:

| Branch | Topic | Description |
|---|---|---|
| **`feat/01-initial-repo-structure`**  | **Tooling & Architecture** | Modern Python setup (`uv`), strict linting, type checks, pre-commit, and CI/CD. |
| `feat/02-forward-graphs` | **Computational Graphs**  | DAG node abstractions, forward evaluation, and graph visualization. |
| `feat/03-scalar-autograd` | **Scalar Autograd**  | Reverse-mode automatic differentiation on scalar values (micrograd style). |
| `feat/04-tensor-autograd` | **Tensor Autograd & VJP**   *(Current)* | Multidimensional arrays, broadcasting, and Vector-Jacobian Products. |
| `feat/05-nn-framework` | **Neural Network Abstractions** | High-level API (`Module`, `Parameter`, `Loss`, `Optimizer`, and `Dataset`). |
| `feat/06-hardware-acceleration` | **GPU Acceleration** | Custom compute kernels with Triton / PyCUDA. |
| `main` | **Full Reference Build** | Fully integrated library with completed lectures, benchmarks, and examples. |

---


## Vector-Jacobian Products (VJP)

Computing the explicit Jacobian matrix $J = \frac{\partial Y}{\partial A}$ for a matrix operation like $Y = A @ B$ is computationally prohibitive ($\mathcal{O}(M \cdot N)$ memory).

Reverse-mode autograd evaluates the **Vector-Jacobian Product directly**, multiplying the incoming loss gradient $G = \frac{\partial L}{\partial Y}$ with $J$ without ever instantiating $J$ in memory:

$$\frac{\partial L}{\partial A} = G @ B^T \quad \text{and} \quad \frac{\partial L}{\partial B} = A^T @ G$$

```text
Forward Pass:
  A: (M, K)
  B: (K, N)
  Y = A @ B: (M, N)

Backward Pass (Incoming Gradient G = dL/dY of shape (M, N)):
  dL/dA = G (M, N) @ B^T (N, K) -> (M, K)  [Inner dimension N cancels]
  dL/dB = A^T (K, M) @ G (M, N) -> (K, N)  [Inner dimension M cancels]
```

---

## Automated Reverse Broadcasting (`unbroadcast`)

In modern deep learning architectures, small parameter tensors (such as a 1D bias vector) are implicitly stretched across batch dimensions during the forward pass:

$$\text{Output} = X W + \mathbf{b} \quad \left(X W \in \mathbb{R}^{B \times D}, \; \mathbf{b} \in \mathbb{R}^{D}\right)$$

In the backward pass, incoming gradients match the expanded shape $(B, D)$. The multivariable chain rule requires that **any dimension expanded during the forward pass must be summed together during the backward pass**:

$$\frac{\partial L}{\partial \mathbf{b}} = \sum_{i=1}^{B} G_{i, :}$$

In `utorch`, this is automated centrally in the `backward()` engine via `unbroadcast(grad, target_shape)`:
1. Extra leading dimensions (e.g. batch size) are summed out.
2. Singleton dimensions (size 1) that were expanded are summed with `keepdims=True`.
3. Primitives remain pure mathematical rules without manual broadcasting boilerplate.

---

## Reduction Calculus (`sum` & `mean`)

When reduction operations collapse dimensions in the forward pass, incoming gradients must be **broadcasted back** across the collapsed axes:

* **Summation Backward**:
  $$\frac{\partial}{\partial X_i} \sum_{j} X_j = 1.0 \implies \frac{\partial L}{\partial X} = \text{broadcast\_to}(G, X.\text{shape})$$
* **Mean Backward**:
  $$\frac{\partial}{\partial X_i} \left(\frac{1}{N} \sum_{j=1}^{N} X_j\right) = \frac{1}{N} \implies \frac{\partial L}{\partial X} = \frac{1}{N} \cdot \text{sum\_backward}(G, X, \text{axis})$$

---


## Lecture Experiments & Highlights

This branch includes an interactive lecture notebook (`examples/04_tensor_autograd.ipynb`) focused on two core pedagogical exercises:

### 1. Dense Layer Projection with Bias Broadcasting ($Y = X W + b$)
Demonstrates the fundamental building block of deep learning:
```python
import utorch as ut

# Inputs: Batch of 4 samples, 3 features (4, 3)
X = ut.Tensor([[1.0, 2.0, 3.0]] * 4, name="X")
# Weights: 3 inputs -> 2 outputs (3, 2)
W = ut.Tensor([[0.5, -0.5], [1.0, 0.0], [-1.0, 1.0]], name="W")
# Bias: 1D vector (2,) broadcasted across all 4 samples
b = ut.Tensor([10.0, -10.0], name="b")

# Forward: (4, 3) @ (3, 2) + (2,) -> (4, 2)
Y = (X @ W) + b
loss = Y.sum()
loss.backward()

assert X.grad.shape == (4, 3)  # G @ W^T
assert W.grad.shape == (3, 2)  # X^T @ G
assert b.grad.shape == (2,)  # unbroadcast(G, (2,)) -> [4.0, 4.0]
```

### 2. Manual Calculus vs. `utorch` Autograd (MSLE Benchmark)
Compares the tedious pen-and-paper derivative derivations from classic lab exercises against automatic differentiation on Mean Squared Logarithmic Error:
$$\text{MSLE} = \frac{1}{B} \sum_{i=1}^{B} \Big(\ln(1 + y_i) - \ln(1 + \hat{y}_i)\Big)^2$$

* **Manual NumPy**: Requires 15 lines of quotient rule derivations, manual transpositions ($X^T @ d_{\text{preds}}$), and explicit `np.sum(..., axis=0)` reductions.
* **`utorch`**: Evaluates the loss directly in mathematical notation; calling `loss.backward()` matches the hand-derived analytical formulas down to $10^{-7}$ precision.

---

##  Getting Started

### Prerequisites
* **Python**: `3.13` or newer
* **uv**: Install via official script:
  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh

  # Windows
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### Local Setup in 3 Commands

1. **Clone the repository and switch to Branch 1**:
   ```bash
   git clone https://github.com/adendek/utorch.git
   cd utorch
   git checkout feat/01-initial-repo-structure
   ```

2. **Sync the virtual environment and dependencies**:
   ```bash
   uv sync
   ```

3. **Install Git pre-commit hooks**:
   ```bash
   uv run pre-commit install --install-hooks
   ```

---

##  Development Command Reference

All commands run through `uv run` to guarantee they execute within the isolated project environment:

| Task | Command | Description |
|---|---|---|
| **Run All Hooks** | `uv run pre-commit run --all-files` | Manually triggers all pre-commit checks |
| **Serve Docs** | `uv run mkdocs serve` | Starts local documentation server at `http://127.0.0.1:8000` |
| **Run Tests** | `uv run pytest` | Executes all unit tests with verbose output |
| **Test Coverage** | `uv run pytest --cov=utorch` | Runs tests and prints coverage table |
| **Lint Code** | `uv run ruff check .` | Runs Ruff linter checks |
| **Auto-fix Lint** | `uv run ruff check . --fix` | Automatically fixes safe linting errors |
| **Format Code** | `uv run ruff format .` | Formats all code according to project style |
| **Type Check** | `uv run mypy utorch/` | Runs strict static type validation |
