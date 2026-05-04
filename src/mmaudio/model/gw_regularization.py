"""
Gromov-Wasserstein regularization for MMAudio.

Enforces geometric alignment between the video conditioning space and the
audio representation space. Implements entropic GW via Sinkhorn iterations,
computed in fp32 for numerical stability.
"""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def pairwise_distances(X: Tensor) -> Tensor:
    """Squared L2 pairwise distance matrix.
    X: (B, D) -> (B, B)
    """
    # ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>
    sq = (X * X).sum(-1, keepdim=True)  # (B, 1)
    D = sq + sq.transpose(0, 1) - 2.0 * (X @ X.transpose(0, 1))
    return D.clamp_min(0.0)


def batched_pairwise_distances(X: Tensor) -> Tensor:
    """Squared L2 pairwise distance matrix, vectorised over a leading batch dim.
    X: (B, k, D) -> (B, k, k)
    """
    sq = (X * X).sum(-1, keepdim=True)  # (B, k, 1)
    D = sq + sq.transpose(1, 2) - 2.0 * torch.bmm(X, X.transpose(1, 2))
    return D.clamp_min(0.0)


def _normalize_dist(D: Tensor) -> Tensor:
    # scale-invariance: divide by mean to avoid lambda_gw having to track feature magnitudes
    m = D.mean().detach().clamp_min(1e-8)
    return D / m


def anticollapse_logdet(v: Tensor, eta: float = 1e-3) -> Tensor:
    """Spectral anti-collapse penalty: ``-log det(K + eta * I)`` on the
    centered Gram matrix ``K = (v - mean) (v - mean)^T / (B - 1)``.

    Non-zero eigenvalues of the Gram matrix coincide with those of the
    feature covariance (up to a (D-B)*log(eta) constant when D > B), so
    using the (B, B) Gram is equivalent to the (D, D) covariance form
    promised in the report and stays cheap when D >> B.

    Rank-1 collapse drives all but one singular value to zero, sending the
    penalty toward (B-1) * (-log eta) -- a large positive number that
    repels the optimiser from the degenerate solution.
    """
    B = v.shape[0]
    if B < 2:
        return torch.zeros((), device=v.device, dtype=v.dtype)
    v_c = v - v.mean(dim=0, keepdim=True)
    K = (v_c @ v_c.transpose(0, 1)) / float(max(B - 1, 1))
    K = K + eta * torch.eye(B, device=v.device, dtype=v.dtype)
    _, logdet = torch.linalg.slogdet(K)
    return -logdet


def entropic_gw_loss(
    D_video: Tensor,
    D_audio: Tensor,
    num_iter: int = 5,
    epsilon: float = 0.1,
    inner_sinkhorn_iter: int = 20,
) -> tuple[Tensor, Tensor]:
    """Approximate entropic Gromov-Wasserstein distance via mirror-descent /
    Sinkhorn projections on the quadratic GW objective.

    D_video, D_audio: (B, B) pairwise squared-distance matrices
    Returns (loss, T) where T is the final (B, B) coupling matrix.
    """
    B = D_video.shape[0]
    device = D_video.device
    dtype = D_video.dtype

    # uniform marginals
    p = torch.full((B,), 1.0 / B, device=device, dtype=dtype)
    q = torch.full((B,), 1.0 / B, device=device, dtype=dtype)

    # initialize coupling as independent outer product
    T = p.unsqueeze(1) * q.unsqueeze(0)  # (B, B)

    # Peyre et al. 2016: at each step, linearize the quadratic cost around T
    # cost tensor L_{ijkl} = (D_v[i,k] - D_a[j,l])^2
    # gradient wrt T at T: G = -2 D_v T D_a  (plus constants independent of T)
    # We minimize <G, T'> + eps * H(T') subject to marginals -> Sinkhorn.

    for _ in range(num_iter):
        G = -2.0 * (D_video @ T @ D_audio)
        # log-domain shift for numerical stability: exp(x - max(x)) keeps K in [0,1].
        # The shift cancels in the u/v Sinkhorn iterations.
        logK = -G / epsilon
        K = torch.exp(logK - logK.max())  # (B, B)
        # Sinkhorn projection
        u = torch.ones(B, device=device, dtype=dtype)
        v = torch.ones(B, device=device, dtype=dtype)
        for _ in range(inner_sinkhorn_iter):
            u = p / (K @ v + 1e-8)
            v = q / (K.transpose(0, 1) @ u + 1e-8)
        T = u.unsqueeze(1) * K * v.unsqueeze(0)

    # final GW objective: sum_{ijkl} (D_v[i,k] - D_a[j,l])^2 * T[i,j] * T[k,l]
    #   = <D_v^2, p p^T> + <D_a^2, q q^T> - 2 <D_v T D_a, T>   (after expansion)
    # but with uniform p,q the first two terms are constants wrt optimization;
    # we return the full value so magnitudes are interpretable.
    Dv2 = (D_video * D_video) @ (p.unsqueeze(1) * p.unsqueeze(0)).sum(1, keepdim=True)
    Da2 = (D_audio * D_audio) @ (q.unsqueeze(1) * q.unsqueeze(0)).sum(1, keepdim=True)
    const = Dv2.sum() + Da2.sum()
    cross = (D_video @ T @ D_audio * T).sum()
    loss = const - 2.0 * cross
    return loss, T


def fused_gw_loss(
    D_video: Tensor,
    D_audio: Tensor,
    C_cross: Tensor,
    alpha: float = 0.5,
    num_iter: int = 5,
    epsilon: float = 0.1,
    inner_sinkhorn_iter: int = 20,
) -> tuple[Tensor, Tensor]:
    """Fused GW: combine pointwise cross-domain cost C_cross (B,B) with relational GW.
    alpha=1 -> pure GW; alpha=0 -> pure Wasserstein."""
    B = D_video.shape[0]
    device = D_video.device
    dtype = D_video.dtype

    p = torch.full((B,), 1.0 / B, device=device, dtype=dtype)
    q = torch.full((B,), 1.0 / B, device=device, dtype=dtype)
    T = p.unsqueeze(1) * q.unsqueeze(0)

    for _ in range(num_iter):
        G_gw = -2.0 * (D_video @ T @ D_audio)
        G = alpha * G_gw + (1.0 - alpha) * C_cross
        K = torch.exp(-G / epsilon)
        u = torch.ones(B, device=device, dtype=dtype)
        v = torch.ones(B, device=device, dtype=dtype)
        for _ in range(inner_sinkhorn_iter):
            u = p / (K @ v + 1e-8)
            v = q / (K.transpose(0, 1) @ u + 1e-8)
        T = u.unsqueeze(1) * K * v.unsqueeze(0)

    # GW term
    Dv2 = (D_video * D_video) @ (p.unsqueeze(1) * p.unsqueeze(0)).sum(1, keepdim=True)
    Da2 = (D_audio * D_audio) @ (q.unsqueeze(1) * q.unsqueeze(0)).sum(1, keepdim=True)
    gw = Dv2.sum() + Da2.sum() - 2.0 * (D_video @ T @ D_audio * T).sum()
    w = (C_cross * T).sum()
    loss = alpha * gw + (1.0 - alpha) * w
    return loss, T


def fused_got_loss(
    D_x: Tensor,
    D_y: Tensor,
    C_xy: Tensor,
    alpha: float = 0.5,
    num_iter: int = 5,
    epsilon: float = 0.1,
    inner_sinkhorn_iter: int = 20,
) -> tuple[Tensor, Tensor]:
    """Fused-GW with a single shared transport plan, vectorised over a batch
    dim and supporting non-square couplings.

    Implements eq. (6)/(8) of Chen et al. 2020, "Graph Optimal Transport for
    Cross-Domain Alignment" (ICML), generalised to per-sample token graphs:
    each item carries an n-node graph on one side and an m-node graph on the
    other, and one shared T:(B,n,m) is solved by Sinkhorn on the unified cost.

    Shapes:
      D_x:  (B, n, n) intra-domain distances on side x
      D_y:  (B, m, m) intra-domain distances on side y
      C_xy: (B, n, m) cross-domain pointwise cost (e.g. cosine distance)
    Returns (mean-over-batch loss, final T).
    """
    Bsz, n, _ = D_x.shape
    m = D_y.shape[1]
    device, dtype = D_x.device, D_x.dtype

    p = torch.full((Bsz, n, 1), 1.0 / n, device=device, dtype=dtype)
    q = torch.full((Bsz, m, 1), 1.0 / m, device=device, dtype=dtype)
    T = p * q.transpose(1, 2)  # (B, n, m)

    for _ in range(num_iter):
        G_gw = -2.0 * torch.bmm(torch.bmm(D_x, T), D_y)
        G = alpha * G_gw + (1.0 - alpha) * C_xy
        logK = -G / epsilon
        logK = logK - logK.amax(dim=(1, 2), keepdim=True)
        K = torch.exp(logK)  # (B, n, m)
        u = torch.ones(Bsz, n, 1, device=device, dtype=dtype)
        v = torch.ones(Bsz, m, 1, device=device, dtype=dtype)
        for _ in range(inner_sinkhorn_iter):
            u = p / (torch.bmm(K, v) + 1e-8)
            v = q / (torch.bmm(K.transpose(1, 2), u) + 1e-8)
        T = u * K * v.transpose(1, 2)

    pp = p * p.transpose(1, 2)  # (B, n, n)
    qq = q * q.transpose(1, 2)  # (B, m, m)
    gw = ((D_x * D_x) * pp).sum(dim=(1, 2)) \
        + ((D_y * D_y) * qq).sum(dim=(1, 2)) \
        - 2.0 * (torch.bmm(torch.bmm(D_x, T), D_y) * T).sum(dim=(1, 2))
    w = (C_xy * T).sum(dim=(1, 2))
    per_sample = alpha * gw + (1.0 - alpha) * w
    return per_sample.mean(), T


def _extract_representations(
    network,
    variant: str,
    clip_f_raw: Tensor,
    x1: Tensor,
    sync_f_raw: Optional[Tensor] = None,
    text_f_raw: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor]:
    """Pull (video_repr, audio_repr), each (B, D), according to variant.

    Uses the DDP-unwrapped module (caller must pass `network.module` if DDP).
    """
    if variant == "global":
        # raw CLIP avg-pool vs raw (normalized) x1 avg-pool
        v = clip_f_raw.mean(dim=1)
        a = x1.mean(dim=1)
    elif variant == "projected":
        v = network.clip_input_proj(clip_f_raw).mean(dim=1)
        a = network.audio_input_proj(x1).mean(dim=1)
    elif variant == "c_g":
        # clip_f_c side of the global conditioning vector
        clip_proj = network.clip_input_proj(clip_f_raw)  # (B, 64, D)
        v = network.clip_cond_proj(clip_proj.mean(dim=1))  # (B, D)
        a = network.audio_input_proj(x1).mean(dim=1)
    elif variant == "fused":
        # same reps as projected; cross-domain cost handled by caller
        v = network.clip_input_proj(clip_f_raw).mean(dim=1)
        a = network.audio_input_proj(x1).mean(dim=1)
    else:
        raise ValueError(f"Unknown GW variant: {variant}")
    return v, a


def compute_gw_regularization(
    network,
    *,
    variant: str,
    clip_f_raw: Tensor,
    x1: Tensor,
    video_exist: Tensor,
    detach_video: bool = True,
    num_sinkhorn_iter: int = 5,
    epsilon: float = 0.1,
    alpha: float = 0.5,
    min_batch: int = 4,
    anticollapse_weight: float = 0.0,
    anticollapse_eta: float = 1e-3,
    anticollapse_target: str = "both",
) -> tuple[Tensor, Optional[Tensor], Tensor]:
    """Compute GW loss over the subset of the batch that has real video features.

    Returns (loss, T, anticollapse_term). Returns (zero, None, zero) if the
    real-video subset is too small. Runs in fp32 inside an
    autocast(enabled=False) block for numerical stability.

    ``loss`` already includes ``anticollapse_weight * anticollapse_term``;
    the third value is returned separately for logging.

    The anti-collapse penalty is applied on the *non-detached* representations
    so that gradient always reaches the learnable projector(s) -- otherwise
    ``detach_video=True`` would defeat its purpose.
    """
    device = x1.device
    zero = torch.zeros((), device=device, dtype=torch.float32)

    # subset to real video samples
    idx = video_exist.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() < min_batch:
        return zero, None, zero

    clip_sub = clip_f_raw.index_select(0, idx)
    x1_sub = x1.index_select(0, idx)

    with torch.cuda.amp.autocast(enabled=False):
        clip_sub = clip_sub.float()
        x1_sub = x1_sub.float()

        # got_token: per-sample fused-GW on post-projection token graphs
        # (Chen et al. 2020, GOT). Aligns intra-video token relations to
        # intra-audio token relations *within each sample*, with one shared T.
        if variant == "got_token":
            v_tok = network.clip_input_proj(clip_sub)         # (B, n, D)
            a_tok = network.audio_input_proj(x1_sub)          # (B, m, D)
            if detach_video:
                v_tok = v_tok.detach()

            D_x = batched_pairwise_distances(v_tok)
            D_y = batched_pairwise_distances(a_tok)
            # per-sample mean normalisation -> lambda_gw is scale-invariant
            D_x = D_x / D_x.mean(dim=(1, 2), keepdim=True).detach().clamp_min(1e-8)
            D_y = D_y / D_y.mean(dim=(1, 2), keepdim=True).detach().clamp_min(1e-8)
            v_n = F.normalize(v_tok, dim=-1)
            a_n = F.normalize(a_tok, dim=-1)
            C_xy = 1.0 - torch.bmm(v_n, a_n.transpose(1, 2))  # (B, n, m)

            gw_loss, T = fused_got_loss(
                D_x, D_y, C_xy,
                alpha=alpha,
                num_iter=num_sinkhorn_iter,
                epsilon=epsilon,
            )
            if not torch.isfinite(gw_loss):
                return zero, None, zero
            # anti-collapse not applied: token graphs don't suffer the
            # rank-1 batch collapse the penalty was designed to repel.
            return gw_loss, T, zero

        v, a = _extract_representations(network, variant, clip_sub, x1_sub)

        # spectral anti-collapse penalty on the un-detached projector outputs
        ac = zero
        if anticollapse_weight > 0.0:
            if anticollapse_target in ("video", "both"):
                ac = ac + anticollapse_logdet(v, eta=anticollapse_eta)
            if anticollapse_target in ("audio", "both"):
                ac = ac + anticollapse_logdet(a, eta=anticollapse_eta)

        v_gw = v.detach() if detach_video else v

        D_v = _normalize_dist(pairwise_distances(v_gw))
        D_a = _normalize_dist(pairwise_distances(a))

        if variant == "fused":
            # cosine-based cross-domain cost between projected reps of same pair
            v_n = F.normalize(v_gw, dim=-1)
            a_n = F.normalize(a, dim=-1)
            C_cross = 1.0 - v_n @ a_n.transpose(0, 1)
            gw_loss, T = fused_gw_loss(
                D_v, D_a, C_cross,
                alpha=alpha,
                num_iter=num_sinkhorn_iter,
                epsilon=epsilon,
            )
        else:
            gw_loss, T = entropic_gw_loss(
                D_v, D_a,
                num_iter=num_sinkhorn_iter,
                epsilon=epsilon,
            )

        if not torch.isfinite(gw_loss) or not torch.isfinite(ac):
            return zero, None, zero

        loss = gw_loss + anticollapse_weight * ac
    return loss, T, ac.detach()


def lambda_schedule(step: int, *, base: float, warmup_steps: int, schedule: str,
                    total_steps: int) -> float:
    """Compute effective lambda_gw at the given step."""
    if step < warmup_steps:
        if schedule == "linear_rampup":
            return base * (step / max(warmup_steps, 1))
        return 0.0
    if schedule == "constant":
        return base
    if schedule == "linear_rampup":
        return base
    if schedule == "cosine_anneal":
        import math
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return base * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unknown schedule: {schedule}")
