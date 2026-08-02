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
not infer independent batch axes. The implicit backward mode treats the
operator as fixed, so use ``backward_mode="unrolled"`` when operator parameters
require gradients.

FBPD's historical ``G_norm`` argument is the squared norm
:math:`\|G\|_2^2`, not :math:`\|G\|_2`. New code can use the explicit
``G_norm_squared`` keyword. Pass a conservative upper bound (for example, a
slightly inflated squared estimate from ``power_iter``). The historical default
step sizes are on Condat's critical boundary when the bound is exact and
``p=1``; use a strict upper bound, a smaller explicit ``sigma``, or ``p<1`` for
the theorem's strict relaxation margin. The result state includes the dual
iterate for a later warm start.


.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.alg.CG
   mirtorch.alg.POGM
   mirtorch.alg.FBPD
   mirtorch.alg.FISTA
   mirtorch.alg.SolverResult
   mirtorch.alg.power_iter
