import math
from typing import Iterable, List, Sequence

import torch

from .schemas import TokenAttribution


SUPPORTED_ATTENTION_STRATEGIES = (
    "last_layer_cls_mean",
    "all_layers_cls_mean",
    "attention_rollout",
)


def aggregate_cls_attention(attentions: Iterable[torch.Tensor], strategy: str) -> torch.Tensor:
    """Return one CLS-to-position score per batch item and sequence position."""
    layers = tuple(attentions or ())
    if not layers:
        raise ValueError("missing_attention_tensors")
    if strategy not in SUPPORTED_ATTENTION_STRATEGIES:
        raise ValueError(f"unsupported_attention_strategy:{strategy}")

    for layer in layers:
        if layer.ndim != 4:
            raise ValueError("invalid_attention_shape")

    if strategy == "last_layer_cls_mean":
        return layers[-1][:, :, 0, :].mean(dim=1)

    if strategy == "all_layers_cls_mean":
        per_layer = [layer[:, :, 0, :].mean(dim=1) for layer in layers]
        return torch.stack(per_layer, dim=0).mean(dim=0)

    # Attention rollout: average heads, add residual identity, row-normalize,
    # and multiply layer transition matrices from bottom to top.
    batch_size, _, sequence_length, _ = layers[0].shape
    identity = torch.eye(
        sequence_length,
        dtype=layers[0].dtype,
        device=layers[0].device,
    ).unsqueeze(0).expand(batch_size, -1, -1)
    rollout = identity
    for layer in layers:
        transition = layer.mean(dim=1) + identity
        transition = transition / transition.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout = torch.bmm(transition, rollout)
    return rollout[:, 0, :]


def rank_code_tokens(
        input_tokens: Sequence[str],
        input_ids: Sequence[int],
        sequence_regions: Sequence[str],
        token_scores: Sequence[float],
        strategy: str,
        top_k: int = None) -> List[TokenAttribution]:
    """Rank observed added/removed token occurrences within one commit."""
    if len(input_ids) != len(sequence_regions) or len(input_ids) != len(token_scores):
        raise ValueError("token_alignment_length_mismatch")

    candidates = []
    for position, region in enumerate(sequence_regions):
        if region not in {"added", "removed"}:
            continue
        score = float(token_scores[position])
        if not math.isfinite(score):
            raise ValueError("non_finite_attention_score")
        if position >= len(input_tokens):
            raise ValueError("observed_token_missing_from_input_tokens")
        candidates.append({
            "sequence_position": position,
            "token": input_tokens[position],
            "token_id": int(input_ids[position]),
            "change_type": region,
            "raw_score": score,
        })

    candidates.sort(key=lambda item: (-item["raw_score"], item["sequence_position"]))
    if candidates:
        minimum = min(item["raw_score"] for item in candidates)
        maximum = max(item["raw_score"] for item in candidates)
        span = maximum - minimum
    else:
        minimum = 0.0
        span = 0.0

    ranked = []
    for rank, item in enumerate(candidates, start=1):
        normalized = 0.0 if span <= 1e-12 else (item["raw_score"] - minimum) / span
        ranked.append(TokenAttribution(
            rank=rank,
            sequence_position=item["sequence_position"],
            token=item["token"],
            token_id=item["token_id"],
            change_type=item["change_type"],
            raw_score=item["raw_score"],
            normalized_score=normalized,
            layer_strategy=strategy,
        ))

    return ranked if top_k is None else ranked[:top_k]
