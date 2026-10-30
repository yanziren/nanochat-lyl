"""
Test the combined MuonAdamW optimizer on CPU.
The AdamW half is checked against torch.optim.AdamW as a reference;
the Muon half is checked behaviorally (determinism + convergence).

python -m pytest tests/test_optim.py -v
"""

import torch
from nanochat.optim import MuonAdamW

DEVICE = "cpu"

ADAMW_SHAPE = (64, 32)
MUON_WIDE_SHAPE = (32, 64)
MUON_TALL_SHAPE = (64, 32)


def make_params_and_groups(seed=1337, offset=0.0):
    """AdamW params plus wide and tall Muon stacks."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    def rand(shape):
        base = torch.randn(shape, generator=gen, device=DEVICE) * 0.05
        return torch.nn.Parameter(base + offset)
    params = [rand(ADAMW_SHAPE), rand(ADAMW_SHAPE)]
    params += [rand(MUON_WIDE_SHAPE) for _ in range(3)]
    params += [rand(MUON_TALL_SHAPE) for _ in range(2)]
    groups = [
        dict(kind="adamw", params=params[0:2], lr=0.02, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        dict(kind="muon", params=params[2:5], lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0),
        dict(kind="muon", params=params[5:7], lr=0.02, momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=0.0),
    ]
    return params, groups


def test_adamw_matches_torch_reference():
    """Our AdamW step must agree with torch.optim.AdamW (both decoupled wd)."""
    hypers = dict(lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    p_ours = torch.nn.Parameter(torch.randn(64, 32, device=DEVICE))
    p_ref = torch.nn.Parameter(p_ours.detach().clone())
    opt_ours = MuonAdamW([dict(kind="adamw", params=[p_ours], **hypers)])
    opt_ref = torch.optim.AdamW([p_ref], **hypers)
    for step in range(10):
        grad = torch.randn(p_ours.shape, generator=torch.Generator(device=DEVICE).manual_seed(step), device=DEVICE)
        p_ours.grad = grad.clone()
        p_ref.grad = grad.clone()
        opt_ours.step()
        opt_ref.step()
    torch.testing.assert_close(p_ours, p_ref, rtol=1e-5, atol=1e-6)


def test_determinism():
    """Two identical runs must produce bitwise identical parameters."""
    results = []
    for _ in range(2):
        params, groups = make_params_and_groups()
        opt = MuonAdamW(groups)
        for step in range(5):
            gen = torch.Generator(device=DEVICE).manual_seed(step)
            for p in params:
                p.grad = torch.randn(p.shape, generator=gen, device=DEVICE) * 0.01
            opt.step()
        results.append([p.detach().clone() for p in params])
    for pa, pb in zip(*results):
        assert torch.equal(pa, pb)


def test_convergence():
    """Optimizing distance-to-target must actually approach the target."""
    targets_params, _ = make_params_and_groups(seed=999)
    targets = [p.detach().clone() for p in targets_params]
    params, groups = make_params_and_groups(seed=999, offset=0.1) # start offset from the targets
    opt = MuonAdamW(groups)
    def distances():
        return [(p.detach() - t).norm().item() for p, t in zip(params, targets)]
    initial = distances()
    for _ in range(50):
        for p, t in zip(params, targets):
            p.grad = 2 * (p.detach() - t) # gradient of ||p - t||^2
        opt.step()
    final = distances()
    for i, (d0, d1) in enumerate(zip(initial, final)):
        assert d1 < 0.5 * d0, f"param {i} did not converge: {d0:.4f} -> {d1:.4f}"
        assert torch.isfinite(params[i]).all()


def test_muon_update_is_orthogonalized():
    """
    The very first Muon update (zero momentum state) of a full-rank gradient
    should be (near) semi-orthogonal after the polar iteration: its nonzero
    singular values land in a band around 1, rather than being spread out.
    """
    p = torch.nn.Parameter(torch.zeros(MUON_WIDE_SHAPE, device=DEVICE))
    group = dict(kind="muon", params=[p], lr=1.0, momentum=0.0, ns_steps=5, beta2=1.0, weight_decay=0.0)
    opt = MuonAdamW([group])
    p.grad = torch.randn(p.shape, generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)
    opt.step()
    # with lr=1, wd=0: p_new = -update, so the update is just -p
    svals = torch.linalg.svdvals(-p.detach().float())
    # NorMuon variance reduction rescales the magnitude, so normalize by the mean
    svals = svals / svals.mean()
    assert svals.max() / svals.min() < 4.0, f"update far from semi-orthogonal: {svals}"
