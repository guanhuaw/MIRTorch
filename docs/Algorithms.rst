Iterative algorithms
====================

The solvers cover the common convex inverse-problem cases:

.. list-table::
   :header-rows: 1

   * - Problem structure
     - Solver
   * - Hermitian positive-definite linear system
     - :class:`mirtorch.alg.CG`
   * - Smooth data term plus a proximal regularizer
     - :class:`mirtorch.alg.FISTA` or :class:`mirtorch.alg.POGM`
   * - Smooth, proximal, and linearly composed terms
     - :class:`mirtorch.alg.FBPD`

Existing calls continue to return a tensor, or ``(tensor, history)`` when an
evaluation function is configured. Pass ``return_info=True`` to ``run`` for a
:class:`mirtorch.alg.SolverResult` containing convergence diagnostics. FISTA
and POGM support opt-in adaptive restart. Their ``rtol`` and ``atol`` settings,
and those of FBPD, are disabled by default so fixed-iteration differentiable
workflows are unchanged.

FISTA and POGM stop on relative iterate change; FBPD uses the combined primal
and dual fixed-point change. These are practical numerical diagnostics, not
certified optimality gaps. Enabling POGM restart or stopping uses the restartable
POGM' recurrence instead of its fixed-horizon final-step coefficient.

For new CG code, prefer ``rtol`` and ``atol``. The older ``tol`` argument is
retained and compares the *squared* residual to preserve compatibility.
``tol=0`` runs exactly ``max_iter`` steps without a convergence synchronization.
CG treats the full shape declared by the ``LinearMap`` as one vector; it does
not infer independent batch axes.

The default implicit CG backward supports first-order derivatives with respect
to ``b``, treating the operator as fixed. It solves the adjoint system with a
normalized right-hand side, zero absolute tolerance, and relative tolerance
``max(1e-12, 10 * dtype_epsilon)``; a smaller explicit positive ``rtol`` is
respected. This prevents an absolute forward threshold from discarding the
small gradients produced by a mean loss. The backward solve reuses ``max_iter``
and preserves fixed-iteration mode when all stopping thresholds are zero.
Its derivative describes the converged solution, so an insufficient iteration
budget can still limit gradient accuracy. Use ``backward_mode="unrolled"`` to
differentiate the actual truncated iterations, operator parameters, or higher
derivatives. Operator-specific derivative restrictions still apply.

Unrolled CG preserves its forward result but rejects backward through a
zero-residual recurrence when the initial residual depends on differentiable
inputs. This can occur at initialization or after exact convergence: masking
the resulting ``0/0`` coefficient cannot generally define the derivative of
the truncated algorithm. The check is conservative even for an identity
operator. For first-order derivatives of ``b`` with a fixed operator, use the
default implicit mode; use a differentiable direct solve when an exact solution
and parameter or higher derivatives are required. Inference and ``max_iter=0``
are unaffected. Stopping before the singular recurrence with a positive
residual threshold retains the derivative of the stopped map; a relative-only
threshold at ``b=0`` is zero and remains guarded.

FBPD's historical ``G_norm`` argument is the squared norm
:math:`\|G\|_2^2`, not :math:`\|G\|_2`. New code can use the explicit
``G_norm_squared`` keyword. Pass a conservative upper bound (for example, a
slightly inflated squared estimate from ``power_iter``). The historical default
step sizes are on Condat's critical boundary when the bound is exact and
``p=1``; use a strict upper bound, a smaller explicit ``sigma``, or ``p<1`` for
the theorem's strict relaxation margin. The result state includes the dual
iterate for a later warm start.


Numerical validation
--------------------

CI tests linear-map adjoints, proximal identities, solver residuals, and
gradients, including double-precision finite differences. It also executes
``demo_mr_physics.ipynb`` and ``demo_trajectory_optimization.ipynb`` on CPU
against an installed wheel; their numerical and reconstruction assertions
must pass, and executed notebooks are saved as CI artifacts.

Before a release, ``Python-CI`` can be dispatched with ``accelerator=cuda`` or
``accelerator=mps`` on a configured self-hosted runner labeled ``self-hosted``
and ``cuda`` or ``mps``. This runs the test suite and those tutorials plus
``demo_3d.ipynb`` on the selected device. It fails if the device is unavailable;
CUDA also requires working native cuFINUFFT. The default ``none`` does not
schedule an accelerator runner. Set ``MIRTORCH_EXAMPLE_DEVICE`` to ``cpu``,
``cuda``, or ``mps`` to reproduce these notebook runs locally.


.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.alg.CG
   mirtorch.alg.POGM
   mirtorch.alg.FBPD
   mirtorch.alg.FISTA
   mirtorch.alg.SolverResult
   mirtorch.alg.power_iter
