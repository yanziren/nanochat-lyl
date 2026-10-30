"""
A mixed AdamW/Muon optimizer for a single process.
Embeddings and scalars go into AdamW, and the matrix parameters go into Muon.

Adapted from: https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
"""

import torch
from torch import Tensor
from nanochat.common import COMPUTE_DTYPE

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer.
https://arxiv.org/abs/1711.05101
"""

def adamw_step(
    p: Tensor,              # parameter tensor
    grad: Tensor,           # gradient, same shape as p
    exp_avg: Tensor,        # first moment, same shape as p
    exp_avg_sq: Tensor,     # second moment, same shape as p
    step: float,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    wd: float,
) -> None:
    """AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update."""
    # Some params (wte, value_embeds) are stored in bf16, so do the math in fp32 and
    # cast back at the end. MPS errors on mixed-dtype ops (CUDA promotes them), and
    # scalar arithmetic like 1 - beta2 loses all precision in bf16.
    p32 = p.float()
    exp_avg32 = exp_avg.float()
    exp_avg_sq32 = exp_avg_sq.float()
    grad32 = grad.float()
    # Weight decay (decoupled, applied before the update)
    p32.mul_(1 - lr * wd)
    # Update running averages
    exp_avg32.lerp_(grad32, 1 - beta1)
    exp_avg_sq32.lerp_(grad32.square(), 1 - beta2)
    # Bias corrections
    bias1 = 1 - beta1 ** step
    bias2 = 1 - beta2 ** step
    # Compute update and apply
    denom = (exp_avg_sq32 / bias2).sqrt() + eps
    step_size = lr / bias1
    p32.add_(exp_avg32 / denom, alpha=-step_size)
    # Write back (no-ops in the common case where everything is already fp32)
    p.copy_(p32)
    exp_avg.copy_(exp_avg32)
    exp_avg_sq.copy_(exp_avg_sq32)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
https://github.com/KellerJordan/modded-nanogpt

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Two more (very) slight and optional improvements:
1) MuonEq row equilibration: rescale each row to the mean row norm so the spectrum
entering orthogonalization is better conditioned (https://arxiv.org/abs/2603.28254)
2) Muon+ renormalization: snap the Frobenius norm to sqrt(min(m, n)), the norm of an exactly
semi-orthogonal matrix, correcting for under-convergence of the polar iteration (https://arxiv.org/abs/2602.21545)

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def muon_step(
    stacked_grads: Tensor,          # (n, rows, cols) - stacked gradients
    stacked_params: Tensor,         # (n, rows, cols) - stacked parameters
    momentum_buffer: Tensor,        # (n, rows, cols) - first moment buffer
    second_momentum_buffer: Tensor, # (n, rows, 1) or (n, 1, cols) - factored second moment
    momentum: float,
    lr: float,
    wd: float,
    beta2: float,
    ns_steps: int,                  # number of Polar Express iterations
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
    """Muon step: momentum -> polar_express -> variance_reduction -> cautious_update."""

    # Nesterov momentum
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
    X = g.bfloat16() if COMPUTE_DTYPE == torch.bfloat16 else g

    # MuonEq row equilibration: rescale each row to the mean row norm so the spectrum entering orthogonalization is better conditioned
    target = X.float().norm(dim=(-2, -1), keepdim=True) / (X.size(-2) ** 0.5)
    row_norm = X.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
    X = X * (target / row_norm).to(X.dtype)

    # Polar Express orthogonalization: replace each update with the nearest orthogonal matrix
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    # Cast back to the param dtype (MPS errors on the mixed-dtype ops below when X is bf16)
    g = X.to(stacked_params.dtype)

    # Muon+ renormalization: snap Frobenius norm to sqrt(min(m, n))
    target_norm = min(g.size(-2), g.size(-1)) ** 0.5
    current_norm = g.float().norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    g = g * (target_norm / current_norm).to(g.dtype)

    # Variance reduction
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Polar Express iteration.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
              All params in a Muon group must have the same shape.
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})

    def _step_adamw(self, group: dict) -> None:
        for p in group['params']:
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            adamw_step(
                p, grad, state['exp_avg'], state['exp_avg_sq'],
                state['step'], group['lr'], group['betas'][0], group['betas'][1],
                group['eps'], group['weight_decay'],
            )

    def _step_muon(self, group: dict) -> None:
        params = group['params']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        grad_chunk = torch.stack([param.grad for param in params])

        state = self.state[p]
        chunk_size = len(params)
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        stacked_params = torch.stack(params)
        muon_step(
            grad_chunk, stacked_params,
            state["momentum_buffer"], state["second_momentum_buffer"],
            group["momentum"],
            group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5,
            group["weight_decay"],
            group["beta2"],
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")
