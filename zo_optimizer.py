"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args:
        model:            The ``nn.Module`` to optimize.
        lr:               Step size / learning rate.
        eps:              Perturbation magnitude for the finite-difference
                          estimator.
        perturbation_mode: Distribution used to sample the perturbation
                          direction. ``"gaussian"`` draws from N(0, I);
                          ``"uniform"`` draws from U(-1, 1) and normalises.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
           Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
           implement dynamic layer selection strategies.

    Example — tune only the final linear layer::

        optimizer = ZeroOrderOptimizer(model)
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.001,
        eps: float = 1e-3,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode

        # ------------------------------------------------------------------
        # STUDENT: Set self.layer_names to the parameters you want to tune.
        #
        # The default below selects only the final classification head.
        # You may replace this with any subset of named parameters, e.g.:
        #   self.layer_names = ["layer4.1.conv2.weight", "fc.weight", "fc.bias"]
        #
        # You can also update self.layer_names inside .step() to implement
        # a dynamic schedule (e.g. gradually unfreeze deeper layers).
        # ------------------------------------------------------------------

        self.layer_names: list[str] = ["fc.weight", "fc.bias"]
        # self.layer_names: list[str] = ["fc.bias"]

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.adam_eps = 1e-8
        self.step_count = 0
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}
        self.max_update_norm = 1.0
        self.spsa_k = 1
        self.adaptive_eps = False
        self.layerwise_lr = {
        "fc": 1.0,
        }
        self.unfreeze_schedule = [20, 200, 600]
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # -------------------------------------------------------------+-----

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample a Rademacher perturbation tensor for SPSA.
        Each element is independently sampled as either -1 or +1

        Args:
            param: The parameter tensor whose shape determines the output shape.

        Returns:
            A tensor of the same shape as ``param`` with values in {-1, +1}.
        """
        
        return torch.empty_like(param).bernoulli_(0.5).mul_(2.0).sub_(1.0)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate pseudo-gradients with SPSA.

        This method changes all selected parameters at the same time using random -1/+1 directions.

        For each random direction, it:
            1. Computes the loss after a small positive change.
            2. Computes the loss after a small negative change.
            3. Uses the difference between these losses to estimate an update direction.

        If ``self.spsa_k`` is more than 1, the method repeats this process several times and averages 
        the estimates to reduce noise.

        Args:
            loss_fn: Function that returns the loss on the current batch.
            params: Parameters selected for tuning.

        Returns:
            A dictionary with one pseudo-gradient tensor for each parameter.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the gradient estimation below.
        # ------------------------------------------------------------------
        grads_acc: dict[str, torch.Tensor] = {
            name: torch.zeros_like(p) for name, p in params.items()
            }
        
        K = max(1, int(getattr(self, "spsa_k", 1)))

        with torch.no_grad():
            eps_t = self.eps
            if getattr(self, "adaptive_eps", False):
                eps_t = self.eps / math.sqrt(1.0 + float(self.step_count))

            for _ in range(K):
                 # Creating random -1/+1 changes for all selected parameters
                perturbations = {
                    name: self._sample_direction(param) for name, param in params.items()
                    }

                # Checking the loss after a small pos change
                for name, param in params.items():
                    param.data.add_(eps_t * perturbations[name])
                f_plus = loss_fn()

                # Checking the loss after a small neg change
                for name, param in params.items():
                    param.data.sub_(2.0 * eps_t * perturbations[name])
                f_minus = loss_fn()

                # How useful this random direction was
                for name, param in params.items():
                    param.data.add_(eps_t * perturbations[name])

                # Adding this estimate to each selected parameter
                coeff = (f_plus - f_minus) / (2.0 * eps_t)
                for name in params.keys():
                    grads_acc[name].add_(coeff * perturbations[name])

        # Averaging the estimates from all random directions
        grads = {name: grads_acc[name].div_(float(K)) for name in grads_acc.keys()}
        return grads
        # ------------------------------------------------------------------


    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Update selected parameters using Adam-style.

        This update keeps moving averages of:
            1. the pseudo-gradient itself.
            2. the squared pseudo-gradient.

        This helps make noisy zero-order updates more stable.

        Args:
            params: Parameters selected for tuning.
            grads: Pseudo-gradients estimated for these parameters.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the parameter update below.
        # ------------------------------------------------------------------
        self.step_count += 1

        with torch.no_grad():
            for name, param in params.items():
                grad = grads[name]

                # If first update then creating tensors
                if name not in self.m:
                    self.m[name] = torch.zeros_like(param)
                    self.v[name] = torch.zeros_like(param)

                # Updating the moving average of pseudo-gradients
                self.m[name].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

                # Updating the moving average of squared pseudo-gradients
                self.v[name].mul_(self.beta2).add_(grad.pow(2), alpha=1 - self.beta2)

                # Correcting the moving averages at early steps
                m_hat = self.m[name] / (1 - self.beta1 ** self.step_count)
                v_hat = self.v[name] / (1 - self.beta2 ** self.step_count)

                # An adaptive update direction
                update = m_hat / (v_hat.sqrt() + self.adam_eps) 

                # Limiting very large updates for stability
                update_norm = update.norm()
                if update_norm > self.max_update_norm:
                    update = update * (self.max_update_norm / update_norm)

                # Using a layer-specific learning rate if needed
                prefix = name.split(".")[0] if "." in name else name
                lw = float(self.layerwise_lr.get(prefix, 1.0))
                effective_lr = self.lr * float(self.layerwise_lr.get(prefix, 1.0))

                param.data.sub_(effective_lr * update)
            # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        This method estimates pseudo-gradients with loss function calls and then updates
        only the selected parameters.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        self.layer_names = ["fc.weight", "fc.bias"]
        # self.layer_names = ["fc.weight"]
        # self.layer_names = ["fc.bias"]
            
        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)
