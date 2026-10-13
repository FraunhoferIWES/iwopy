# ADR-0014: Sparse Jacobians And Precise Derivative Terminology

- Status: Accepted
- Date: 2026-10-13
- Supersedes: Derivative-matrix terminology in ADR-0005, ADR-0007, ADR-0008,
  and ADR-0011; their algorithm, batching, and workflow decisions remain in force.

## Context

Dense component-by-variable Jacobians and optimizer caches store structural
zeros unnecessarily. Calling these matrices gradients also obscures the
distinction between a scalar gradient vector and a vector-function Jacobian.
SciPy and PyGMO expose different native derivative contracts.

## Decision

Use `Problem.get_jacobian()` and `calc_jacobian()` for component-by-variable
SciPy CSR matrices. Remove the former plural-gradient matrix APIs without
aliases, updating maintained callers, tests, examples, and documentation.
Reserve gradient terminology for scalar-function vectors and established
gradient-based algorithms; retain backend-required names such as PyGMO's
`gradient`, `has_gradient`, and `gradient_sparsity` methods.
Accepted historical ADRs retain their original wording; this record supersedes
their matrix, cache, and derivative-batching terminology.

Assemble analytical entries directly in sparse storage. Omit exact zeros and
retain unavailable partial derivatives as explicit NaNs until LocalFD resolves
them or validation rejects them. Numerical filling must preserve analytical
entries, independent pairs, mappings, and selected row/column order.

Cache CSR Jacobians in SciPy and SLSQP. GG uses lazy CSR ring-buffer slots and
sparse constraint predictions. Convert only requested rows at dense boundaries.
SciPy trust-constr receives CSR constraint callbacks with `sparse_jacobian=True`
by default; explicit native values remain authoritative. SLSQP requires dense
constraint arrays, and scalar objective gradients remain dense vectors. PyGMO
extracts its sparse native fitness-Jacobian entries directly.

## Consequences

- Sparse Jacobian payload and cache storage scale with nonzero entries rather
  than all component-variable pairs. Dense Jacobians can cost more in CSR.
- Dependency masks and finite-difference stencil/value buffers remain dense.
  Native Hessian approximations and solver workspaces can still dominate memory.
- Dense-only trust-constr factorizations require `sparse_jacobian=False`.
  Native sparse factorizations may still densify when necessary.
- Existing long core and optimizer modules remain under their established
  ownership. Multi-step sparse assembly and finite-difference routines exceed
  the 40-line guideline to keep ordering and numerical state transitions local;
  unrelated size-only refactoring is outside this change.

## Alternatives Considered

- Sparse caches with dense core assembly: rejected because it retains the
  largest avoidable allocation and obscures the public matrix contract.
- Dense conversion for every SciPy method: rejected because trust-constr
  supports sparse constraint Jacobians natively.
- Renaming PyGMO's native methods: rejected because its protocol requires them.
- Legacy matrix aliases: rejected under the forward-only development policy.

## References

- [Architecture](../architecture.md#derivatives)
- [Naming conventions](../naming-conventions.md#bounds-feasibility-and-derivatives)
- [Core API](../source/api_core.rst)
- [SciPy API](../source/api_interfaces_scipy.rst)
