# Contributing to utorch

Thank you for contributing to **utorch**!

utorch is a minimalist, educational implementation of automatic differentiation and deep learning primitives designed for teaching PyTorch from scratch. To keep the codebase clean, readable, and pedagogically sound, please follow these guidelines.

---

## 1. Branch Strategy and Roadmap

`utorch` follows a staged, branch-based progression corresponding to the lecture modules:

| Branch | Topic |
|---|---|
| `main` | Production-ready reference build with all lecture branches integrated |
| `feat/01-initial-repo-structure` | Project layout, build system, linting, and test harness |
| `feat/02-forward-graphs` | Computational graph construction and forward evaluation |
| `feat/03-scalar-autograd` | Scalar reverse-mode automatic differentiation (micrograd-style) |
| `feat/04-tensor-autograd` | Vector-Jacobian products (VJP) and n-dimensional tensors |
| `feat/05-nn-framework` | High-level API (`Module`, `Parameter`, `Loss`, `Optimizer`, `Dataset`) |
| `feat/06-hardware-acceleration` | Custom GPU kernels with Triton / PyCUDA |

> **Note**: Target your work to the specific lecture branch rather than committing directly to `main`.

---

## 2. Development Workflow

### Step 1: Clone the Repository
```bash
git clone https://github.com/adendek/utorch.git
cd utorch
```

### Step 2: Set Up the Environment
We use [`uv`](https://docs.astral.sh/uv/) for fast, deterministic dependency management:

```bash
# Install virtual environment and dependencies
uv sync

# Install git pre-commit hooks
uv run pre-commit install
```

### Step 3: Open an Issue First
Before submitting non-trivial changes, [open an issue](https://github.com/adendek/utorch/issues) describing your proposal or bug fix to ensure alignment with the lecture syllabus.

### Step 4: Create a Dedicated Branch
Create a branch with a descriptive name:

```bash
git checkout -b feat/my-focused-improvement
```
Keep your changes focused on a single responsibility with minimal touched files.

### Step 5: Format and Lint
Ensure formatting and type checks pass:

```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files
```

Expected output:

```
adendek@Mac-JQDQJFRQ1P utorch % uv run pre-commit run --all-files

trim trailing whitespace.................................................Passed
fix end of files.........................................................Passed
check yaml...............................................................Passed
check toml...............................................................Passed
check for added large files..............................................Passed
ruff check...............................................................Passed
ruff format..............................................................Passed
mypy.....................................................................Passed
pytest...................................................................Passed
```

### Step 7: Open a Pull Request
1. Push your branch to GitHub.
2. Open a Pull Request against the relevant lecture branch.
3. Link the PR to your issue (e.g., `Closes #1`).

---

## 3. Educational Code Standards

Because `utorch` is built for teaching:

1. **Clarity over Cleverness**: Write transparent, idiomatic Python. Avoid premature optimization or cryptic metaprogramming unless explicitly covering framework internals.
2. **Explicit Mathematical Documentation**: Differentiable operations should include docstrings explaining the forward formula and the backward gradient derivation (Vector-Jacobian Product).
3. **Numerical Verification**: Every custom autograd operation must include finite-difference gradient check tests comparing analytical gradients against numerical approximations.
4. **Type Annotations**: Use standard Python type hints for all public function signatures.
```
