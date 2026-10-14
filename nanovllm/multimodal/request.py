from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class ImagePrompt:
    """One-image request. Text should contain the model's image placeholder."""

    text: str
    image: Any


@dataclass(slots=True)
class ImageState:
    """Processor-derived immutable image metadata plus cached visual features."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    rope_positions: tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
    ]
    rope_delta: int
    image_start: int
    image_end: int
    grid_signature: tuple[int, int, int]
    visual_features: torch.Tensor | None = None


def _build_rope_positions(
    token_types: list[int],
    grid: tuple[int, int, int],
    spatial_merge_size: int,
) -> tuple[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    int,
]:
    if spatial_merge_size <= 0:
        raise ValueError("spatial_merge_size must be positive")

    positions: list[list[int]] = [[], [], []]
    current_pos = 0
    cursor = 0
    while cursor < len(token_types):
        modality = token_types[cursor]
        end = cursor + 1
        while end < len(token_types) and token_types[end] == modality:
            end += 1

        length = end - cursor
        if modality == 0:
            values = list(range(current_pos, current_pos + length))
            for axis in positions:
                axis.extend(values)
            current_pos += length
        elif modality == 1:
            grid_t, grid_h, grid_w = grid
            llm_t = grid_t
            llm_h = grid_h // spatial_merge_size
            llm_w = grid_w // spatial_merge_size
            if llm_t * llm_h * llm_w != length:
                raise ValueError(
                    f"image placeholder count ({length}) does not match merged grid ({llm_t * llm_h * llm_w})"
                )
            for t in range(llm_t):
                for h in range(llm_h):
                    for w in range(llm_w):
                        positions[0].append(current_pos + t)
                        positions[1].append(current_pos + h)
                        positions[2].append(current_pos + w)
            current_pos += max(grid_h, grid_w) // spatial_merge_size
        else:
            raise ValueError(
                f"ImagePrompt supports text and one image only; found modality type {modality}"
            )
        cursor = end

    rope_positions = tuple(tuple(axis) for axis in positions)
    max_position = max(max(axis) for axis in rope_positions)
    return rope_positions, max_position + 1 - len(token_types)


def image_state_from_processor_output(
    output: Mapping[str, Any],
    root_config: Any,
) -> tuple[list[int], ImageState]:
    required = ("input_ids", "pixel_values", "image_grid_thw", "mm_token_type_ids")
    missing = [name for name in required if name not in output]
    if missing:
        raise ValueError("processor output is missing: " + ", ".join(missing))

    input_ids = output["input_ids"]
    token_types = output["mm_token_type_ids"]
    pixel_values = output["pixel_values"]
    image_grid_thw = output["image_grid_thw"]

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("ImagePrompt processor output must have batch size 1")
    if token_types.shape != input_ids.shape:
        raise ValueError("mm_token_type_ids must match input_ids shape")
    if image_grid_thw.ndim != 2 or image_grid_thw.shape != (1, 3):
        raise ValueError("ImagePrompt requires exactly one image grid")

    ids = [int(value) for value in input_ids[0].tolist()]
    types = [int(value) for value in token_types[0].tolist()]
    image_indices = [index for index, modality in enumerate(types) if modality == 1]
    if not image_indices:
        raise ValueError("processor produced no image feature tokens")
    image_start = image_indices[0]
    image_end = image_indices[-1] + 1
    if image_indices != list(range(image_start, image_end)):
        raise ValueError("ImagePrompt requires one contiguous image interval")
    if any(modality not in (0, 1) for modality in types):
        raise ValueError("ImagePrompt does not support video tokens")

    grid_signature = tuple(int(value) for value in image_grid_thw[0].tolist())
    spatial_merge_size = int(root_config.vision_config.spatial_merge_size)
    expected_visual_tokens = (
        grid_signature[0] * grid_signature[1] * grid_signature[2] // (spatial_merge_size**2)
    )
    if image_end - image_start != expected_visual_tokens:
        raise ValueError(
            f"image placeholder count ({image_end - image_start}) does not match "
            f"visual token count ({expected_visual_tokens})"
        )

    rope_positions, rope_delta = _build_rope_positions(
        types,
        grid_signature,
        spatial_merge_size,
    )
    state = ImageState(
        pixel_values=pixel_values.detach().cpu(),
        image_grid_thw=image_grid_thw.detach().cpu(),
        rope_positions=rope_positions,
        rope_delta=rope_delta,
        image_start=image_start,
        image_end=image_end,
        grid_signature=grid_signature,
    )
    return ids, state
