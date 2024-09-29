# Contributing to MIRTorch

Thank you for your interest in contributing to MIRTorch! MIRTorch is an academic project focused on advancing medical imaging reconstruction methods. Your contributions are valuable in improving and extending this library, e.g., adding new linear operators for modalities beyond MRI, optimization solvers, and other useful tools.

## How to Contribute

We encourage contributions that expand the functionality of MIRTorch while maintaining user-friendly and hackable. Below are some suggestions.

### 1. Reporting Bugs

If you find any bugs, please [open an issue](https://github.com/guanhuaw/MIRTorch/issues) on GitHub. When reporting, please include:
- A clear description of the problem.
- The version of MIRTorch, Python, and PyTorch you are using.

### 2. Implementing LinearMap Extensions

For any work related to `LinearMap`, please ensure:
- **Testing Linearity:** All new LinearMap implementations must be tested for linearity. This means that the map should satisfy both additivity and homogeneity properties:
  - Additivity: `A(x1 + x2) = A(x1) + A(x2)`
  - Homogeneity: `A(c * x) = c * A(x)` for any scalar `c`.

Please consider designing tests that explicitly verify these conditions.

### 3. Optimization Solvers

When working on optimization solvers, please make sure:
- The solver is tested on at least **one kind** of `LinearMap` to ensure compatibility and correctness.
- Tests should include convergence criteria and comparisons with baseline solvers when applicable.

Document any assumptions or limitations in the implementation.

### 4. Submitting Pull Requests

When you're ready to submit your code:
- Fork the repository and make your changes in a new branch.
- Ensure your code passes all existing tests and add new tests if necessary using `pytest`.
- Follow the coding style used in the project.
- Open a pull request with a clear description of your changes.

### 5. Coding Standards

- Write easy-to-read and well-documented code. Clear is better than clever.
- Use meaningful variable and function names.
- Ensure your code is modular and maintainable.
- Include comments where necessary to explain complex sections.

### 6. Tests

All new features must include tests. We use [PyTest](https://pytest.org) for testing. Run tests locally before submitting your code to ensure nothing is broken.

---

Thank you again for contributing to MIRTorch. We look forward to working together!
