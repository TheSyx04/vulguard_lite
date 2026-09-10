"""Grad-CAM for the row stage of DeepJIT's hierarchical code CNN."""

from dataclasses import dataclass
from typing import List

import torch

from .schemas import RowAttribution


def _project_receptive_fields(cam: torch.Tensor, kernel_size: int, row_count: int) -> torch.Tensor:
    """Map commit-CNN positions back to input rows using their exact receptive fields."""
    projected = cam.new_zeros((cam.shape[0], row_count))
    coverage = cam.new_zeros((cam.shape[0], row_count))
    for position in range(cam.shape[1]):
        end = min(position + kernel_size, row_count)
        projected[:, position:end] += cam[:, position].unsqueeze(1)
        coverage[:, position:end] += 1
    return projected / coverage.clamp_min(1)


def hierarchical_row_gradcam(model, message, code, target_class=1):
    """Return receptive-field-aware scores for code rows and their tokens."""
    if target_class not in (0, 1):
        raise ValueError("target_class must be 0 or 1")

    activations = []
    gradients = []
    token_activations = []
    token_gradients = []
    handles = []

    def capture(index):
        def hook(_module, _inputs, output):
            activations.append((index, output))
            output.register_hook(lambda gradient: gradients.append((index, gradient)))
        return hook

    for index, layer in enumerate(model.convs_code_file):
        handles.append(layer.register_forward_hook(capture(index)))

    def capture_tokens(index):
        def hook(_module, _inputs, output):
            token_activations.append((index, output))
            output.register_hook(lambda gradient: token_gradients.append((index, gradient)))
        return hook

    for index, layer in enumerate(model.convs_code_line):
        handles.append(layer.register_forward_hook(capture_tokens(index)))

    try:
        model.zero_grad(set_to_none=True)
        output = model(message, code, return_attribution_data=True)
        target = output["logit"].sum()
        if target_class == 0:
            target = -target
        target.backward()

        activation_by_index = dict(activations)
        gradient_by_index = dict(gradients)
        row_count = int(code.shape[1])
        branches = []
        branch_shapes = []
        for index, kernel_size in enumerate(model.params["filter_sizes"]):
            activation = activation_by_index[index]
            gradient = gradient_by_index[index]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1)).squeeze(-1)
            branches.append(_project_receptive_fields(cam, int(kernel_size), row_count))
            branch_shapes.append(list(activation.shape))

        row_scores = torch.stack(branches, dim=0).mean(dim=0)
        token_activation_by_index = dict(token_activations)
        token_gradient_by_index = dict(token_gradients)
        batch_size, row_count, token_count = code.shape
        token_branches = []
        token_branch_shapes = []
        for index, kernel_size in enumerate(model.params["filter_sizes"]):
            activation = token_activation_by_index[index]
            gradient = token_gradient_by_index[index]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activation).sum(dim=1)).squeeze(-1)
            projected = _project_receptive_fields(cam, int(kernel_size), token_count)
            token_branches.append(projected.reshape(batch_size, row_count, token_count))
            token_branch_shapes.append(list(activation.shape))
        token_scores = torch.stack(token_branches, dim=0).mean(dim=0)
        return {
            "probability": output["probability"].detach(),
            "logit": output["logit"].detach(),
            "row_scores": row_scores.detach(),
            "token_scores": token_scores.detach(),
            "branch_activation_shapes": branch_shapes,
            "token_branch_activation_shapes": token_branch_shapes,
        }
    finally:
        for handle in handles:
            handle.remove()


def rank_observed_rows(row_texts: List[str], scores, strategy: str, top_k=None):
    values = [float(score) for score in scores[:len(row_texts)]]
    if not values:
        return []
    if not all(torch.isfinite(torch.tensor(values))):
        raise ValueError("non_finite_gradcam_score")
    minimum, maximum = min(values), max(values)
    scale = maximum - minimum
    ranked = sorted(enumerate(values), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ranked = ranked[:top_k]
    return [
        RowAttribution(
            rank=rank,
            row_position=position,
            text=row_texts[position],
            raw_score=value,
            normalized_score=(value - minimum) / scale if scale > 0 else 0.0,
            receptive_field_strategy=strategy,
        )
        for rank, (position, value) in enumerate(ranked, 1)
    ]
