# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""3D RoPE (Rotary Position Embedding) utilities for Qwen3-VL MoT.

Qwen3-VL uses MRoPE (Multi-dimensional Rotary Position Embedding) with
head_dim split into 3 segments [24, 20, 20] corresponding to [T, H, W].

For the canonical TRELLIS.2 stage path we use:
- ``ss_latent``: dense raster ``XYZ`` order
- ``shape_slat``: sparse runtime ``XYZ`` coords order
- ``tex_slat``: sparse runtime ``XYZ`` coords order

Key design:
- Text tokens: T=H=W=position (all dimensions same)
- Image tokens: T=offset, H=row+offset, W=col+offset (2D spatial with fixed T)
- Video tokens: T=frame+offset, H=row+offset, W=col+offset (3D temporal+spatial)
- Stage tokens: T=X+offset, H=Y+offset, W=Z+offset

The offset preserves sequence order within each logical sample while keeping
the stage-local 3D geometry in the RoPE axes.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..stages.sparse_bridge import is_trellis2_sparse_tensor


def build_dense_raster_position_ids(
    grid_thw: Union[torch.Tensor, Sequence[int]],
    t_offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Build 3D position IDs for a dense raster stage in native raster order."""
    grid = torch.as_tensor(grid_thw, device=device, dtype=torch.long)
    if grid.ndim != 1 or grid.numel() != 3:
        raise ValueError(f"grid_thw must have shape [3], got {tuple(grid.shape)}.")
    grid_x, grid_y, grid_z = torch.meshgrid(
        torch.arange(int(grid[0].item()), device=device, dtype=torch.long),
        torch.arange(int(grid[1].item()), device=device, dtype=torch.long),
        torch.arange(int(grid[2].item()), device=device, dtype=torch.long),
        indexing="ij",
    )
    offset = torch.tensor(int(t_offset), device=device, dtype=torch.long)
    return torch.stack(
        [
            grid_x.flatten() + offset,
            grid_y.flatten() + offset,
            grid_z.flatten() + offset,
        ],
        dim=0,
    )


def build_sparse_position_ids_from_coords(
    coords: torch.Tensor,
    t_offset: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build 3D position IDs directly from sparse XYZ coords without reordering."""
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape [N, 3], got {tuple(coords.shape)}.")
    resolved_device = device if device is not None else coords.device
    coords = coords.to(device=resolved_device, dtype=torch.long)
    offset = torch.tensor(int(t_offset), device=resolved_device, dtype=torch.long)
    return torch.stack(
        [
            coords[:, 0] + offset,
            coords[:, 1] + offset,
            coords[:, 2] + offset,
        ],
        dim=0,
    )


def compute_mrope_position_deltas(
    position_ids: torch.Tensor,
    *,
    max_len: int,
    attention_mask: Optional[torch.Tensor] = None,
    sample_lens: Optional[List[List[int]]] = None,
) -> torch.Tensor:
    """Compute Qwen-style MRoPE deltas from explicit position IDs."""
    if position_ids.ndim != 3 or position_ids.shape[0] != 3:
        raise ValueError(f"position_ids must have shape [3, B, L], got {tuple(position_ids.shape)}.")
    batch_size = position_ids.shape[1]
    deltas: List[int] = []
    for batch_index in range(batch_size):
        if sample_lens is not None:
            valid_len = int(sum(int(length) for length in sample_lens[batch_index]))
        elif attention_mask is not None:
            valid_len = int(attention_mask[batch_index].sum().item())
        else:
            valid_len = max_len
        if valid_len <= 0:
            deltas.append(-max_len)
            continue
        valid_position_ids = position_ids[:, batch_index, :valid_len]
        deltas.append(int(valid_position_ids.max().item()) + 1 - max_len)
    return torch.tensor(deltas, device=position_ids.device, dtype=position_ids.dtype).unsqueeze(1)


def _normalize_dense_grid_list(
    grid_thw: Optional[Union[torch.Tensor, Sequence[Sequence[int]]]],
) -> List[torch.Tensor]:
    """Normalize dense grid list."""
    if grid_thw is None:
        return []
    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.ndim == 1:
            if grid_thw.numel() != 3:
                raise ValueError(f"Dense grid tensor must have 3 elements, got {tuple(grid_thw.shape)}.")
            return [grid_thw.detach().clone()]
        if grid_thw.ndim == 2 and grid_thw.shape[-1] == 3:
            return [row.detach().clone() for row in grid_thw]
        raise ValueError(f"Dense grid tensor must have shape [3] or [N, 3], got {tuple(grid_thw.shape)}.")

    normalized: List[torch.Tensor] = []
    for item in grid_thw:
        tensor = torch.as_tensor(item, dtype=torch.long)
        if tensor.ndim != 1 or tensor.numel() != 3:
            raise ValueError(f"Each dense grid entry must have shape [3], got {tuple(tensor.shape)}.")
        normalized.append(tensor)
    return normalized


def _normalize_sparse_coord_list(
    coords: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    *,
    name: str,
) -> List[torch.Tensor]:
    """Normalize sparse coord list."""
    if coords is None:
        return []
    if isinstance(coords, torch.Tensor):
        if coords.ndim != 2 or coords.shape[-1] != 3:
            raise ValueError(f"{name} tensor must have shape [N, 3], got {tuple(coords.shape)}.")
        return [coords.detach().clone()]

    normalized: List[torch.Tensor] = []
    for item in coords:
        tensor = torch.as_tensor(item)
        if tensor.ndim != 2 or tensor.shape[-1] != 3:
            raise ValueError(f"Each {name} entry must have shape [N, 3], got {tuple(tensor.shape)}.")
        normalized.append(tensor.detach().clone())
    return normalized


def _extract_sparse_coords_from_runtime_tensor(
    sparse_tensor: Any,
    *,
    name: str,
) -> List[torch.Tensor]:
    """Extract sparse coords from runtime tensor."""
    if sparse_tensor is None:
        return []
    if not is_trellis2_sparse_tensor(sparse_tensor):
        raise TypeError(f"{name} must be a TRELLIS.2 SparseTensor when provided.")
    _, coord_list = sparse_tensor.to_tensor_list()
    normalized: List[torch.Tensor] = []
    for coords in coord_list:
        tensor = torch.as_tensor(coords)
        if tensor.ndim != 2 or tensor.shape[-1] not in (3, 4):
            raise ValueError(f"Each {name} coord entry must have shape [N, 3] or [N, 4], got {tuple(tensor.shape)}.")
        if tensor.shape[-1] == 4:
            tensor = tensor[:, 1:]
        normalized.append(tensor.to(dtype=torch.long).detach().clone())
    return normalized


def _build_text_segment_position_ids(length: int, start_offset: int, device: torch.device) -> torch.Tensor:
    """Build text segment position ids."""
    if length <= 0:
        return torch.empty((3, 0), device=device, dtype=torch.long)
    start = torch.tensor(int(start_offset), device=device, dtype=torch.long)
    text_positions = torch.arange(length, device=device, dtype=torch.long) + start
    return text_positions.view(1, -1).expand(3, -1)


def _validate_stage_block(
    *,
    input_ids_sample: torch.LongTensor,
    content_start: int,
    content_len: int,
    pad_token_id: int,
    end_token_id: int,
    stage_name: str,
) -> None:
    """Validate stage block."""
    if content_start + content_len >= input_ids_sample.shape[0]:
        raise ValueError(
            f"{stage_name} block overflows sample length: start={content_start}, len={content_len}, "
            f"sample_len={input_ids_sample.shape[0]}."
        )
    pad_slice = input_ids_sample[content_start : content_start + content_len]
    if not torch.all(pad_slice.eq(pad_token_id)):
        raise ValueError(f"{stage_name} block pad tokens do not match the expected placeholder layout.")
    if int(input_ids_sample[content_start + content_len].item()) != end_token_id:
        raise ValueError(f"{stage_name} block is missing the expected end token after the content span.")


def _compute_single_sample_mot_positions(
    *,
    input_ids_sample: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    ss_grid_thw: List[torch.Tensor],
    shape_coord_blocks: List[torch.Tensor],
    tex_coord_blocks: List[torch.Tensor],
    image_index: int,
    video_index: int,
    ss_index: int,
    shape_index: int,
    tex_index: int,
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    ss_start_token_id: Optional[int],
    ss_pad_token_id: Optional[int],
    ss_end_token_id: Optional[int],
    shape_start_token_id: Optional[int],
    shape_pad_token_id: Optional[int],
    shape_end_token_id: Optional[int],
    tex_start_token_id: Optional[int],
    tex_pad_token_id: Optional[int],
    tex_end_token_id: Optional[int],
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Compute MRoPE position IDs for one logical sample, including TRELLIS stage tokens."""
    device = input_ids_sample.device
    sample_len = input_ids_sample.shape[0]
    if sample_len == 0:
        empty = torch.empty((3, 0), device=device, dtype=torch.long)
        return empty, {
            "image_index": image_index,
            "video_index": video_index,
            "ss_index": ss_index,
            "shape_index": shape_index,
            "tex_index": tex_index,
        }

    vision_start_indices = torch.argwhere(input_ids_sample == vision_start_token_id).squeeze(1)
    image_block_starts: List[int] = []
    video_block_starts: List[int] = []
    if vision_start_indices.numel() > 0:
        for index in vision_start_indices.tolist():
            if index + 1 >= sample_len:
                continue
            next_token = int(input_ids_sample[index + 1].item())
            if next_token == image_token_id:
                image_block_starts.append(index + 1)
            elif next_token == video_token_id:
                video_block_starts.append(index + 1)

    def _stage_content_starts(start_token_id: Optional[int], end_token_id: Optional[int]) -> List[int]:
        """Handle stage content starts."""
        if start_token_id is None or end_token_id is None:
            return []
        indices = torch.argwhere(input_ids_sample == start_token_id).squeeze(1)
        content_starts: List[int] = []
        for index in indices:
            start = int(index.item())
            content_start = start + 1
            if content_start < sample_len and int(input_ids_sample[content_start].item()) == int(end_token_id):
                continue
            content_starts.append(content_start)
        return content_starts

    ss_block_starts = _stage_content_starts(ss_start_token_id, ss_end_token_id)
    shape_block_starts = _stage_content_starts(shape_start_token_id, shape_end_token_id)
    tex_block_starts = _stage_content_starts(tex_start_token_id, tex_end_token_id)

    image_ptr = video_ptr = ss_ptr = shape_ptr = tex_ptr = 0
    parts: List[torch.Tensor] = []
    cursor = 0

    while True:
        next_positions = {
            "image": image_block_starts[image_ptr] if image_ptr < len(image_block_starts) else sample_len + 1,
            "video": video_block_starts[video_ptr] if video_ptr < len(video_block_starts) else sample_len + 1,
            "ss": ss_block_starts[ss_ptr] if ss_ptr < len(ss_block_starts) else sample_len + 1,
            "shape": shape_block_starts[shape_ptr] if shape_ptr < len(shape_block_starts) else sample_len + 1,
            "tex": tex_block_starts[tex_ptr] if tex_ptr < len(tex_block_starts) else sample_len + 1,
        }
        next_kind = min(next_positions, key=next_positions.get)
        next_start = next_positions[next_kind]
        if next_start > sample_len:
            break

        text_len = next_start - cursor
        if text_len < 0:
            raise ValueError(f"Invalid multimodal ordering: cursor={cursor}, next_start={next_start}.")
        start_offset = int(parts[-1].max().item()) + 1 if parts else 0
        text_positions = _build_text_segment_position_ids(text_len, start_offset, device)
        if text_positions.numel() > 0:
            parts.append(text_positions)
        content_offset = start_offset + text_len

        if next_kind == "image":
            if image_grid_thw is None:
                raise ValueError("image_grid_thw must be provided when image tokens are present.")
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            image_ptr += 1
            llm_grid_t = int(t.item())
            llm_grid_h = int(h.item()) // spatial_merge_size
            llm_grid_w = int(w.item()) // spatial_merge_size
            t_index = torch.arange(llm_grid_t, device=device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h, device=device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            content_positions = torch.stack([t_index, h_index, w_index]) + content_offset
            parts.append(content_positions)
            cursor = next_start + llm_grid_t * llm_grid_h * llm_grid_w
        elif next_kind == "video":
            if video_grid_thw is None:
                raise ValueError("video_grid_thw must be provided when video tokens are present.")
            t, h, w = video_grid_thw[video_index]
            video_index += 1
            video_ptr += 1
            llm_grid_t = int(t.item())
            llm_grid_h = int(h.item()) // spatial_merge_size
            llm_grid_w = int(w.item()) // spatial_merge_size
            t_index = torch.arange(llm_grid_t, device=device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h, device=device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            content_positions = torch.stack([t_index, h_index, w_index]) + content_offset
            parts.append(content_positions)
            cursor = next_start + llm_grid_t * llm_grid_h * llm_grid_w
        elif next_kind == "ss":
            if ss_index >= len(ss_grid_thw):
                raise ValueError("ss_grid_thw metadata is shorter than the number of ss blocks.")
            grid = ss_grid_thw[ss_index]
            ss_index += 1
            ss_ptr += 1
            content_positions = build_dense_raster_position_ids(grid, content_offset, device)
            _validate_stage_block(
                input_ids_sample=input_ids_sample,
                content_start=next_start,
                content_len=content_positions.shape[1],
                pad_token_id=int(ss_pad_token_id),
                end_token_id=int(ss_end_token_id),
                stage_name="ss",
            )
            parts.append(content_positions)
            cursor = next_start + content_positions.shape[1]
        elif next_kind == "shape":
            if shape_index >= len(shape_coord_blocks):
                raise ValueError("shape_coords metadata is shorter than the number of shape blocks.")
            coords = shape_coord_blocks[shape_index]
            shape_index += 1
            shape_ptr += 1
            content_positions = build_sparse_position_ids_from_coords(coords, content_offset, device=device)
            _validate_stage_block(
                input_ids_sample=input_ids_sample,
                content_start=next_start,
                content_len=content_positions.shape[1],
                pad_token_id=int(shape_pad_token_id),
                end_token_id=int(shape_end_token_id),
                stage_name="shape",
            )
            parts.append(content_positions)
            cursor = next_start + content_positions.shape[1]
        elif next_kind == "tex":
            if tex_index >= len(tex_coord_blocks):
                raise ValueError("tex_coords metadata is shorter than the number of tex blocks.")
            coords = tex_coord_blocks[tex_index]
            tex_index += 1
            tex_ptr += 1
            content_positions = build_sparse_position_ids_from_coords(coords, content_offset, device=device)
            _validate_stage_block(
                input_ids_sample=input_ids_sample,
                content_start=next_start,
                content_len=content_positions.shape[1],
                pad_token_id=int(tex_pad_token_id),
                end_token_id=int(tex_end_token_id),
                stage_name="tex",
            )
            parts.append(content_positions)
            cursor = next_start + content_positions.shape[1]
        else:
            raise AssertionError(f"Unhandled multimodal block kind: {next_kind}")

    if cursor < sample_len:
        start_offset = int(parts[-1].max().item()) + 1 if parts else 0
        trailing_positions = _build_text_segment_position_ids(sample_len - cursor, start_offset, device)
        if trailing_positions.numel() > 0:
            parts.append(trailing_positions)

    combined = torch.cat(parts, dim=1) if parts else torch.empty((3, 0), device=device, dtype=torch.long)
    if combined.shape[1] != sample_len:
        raise ValueError(
            f"Position builder produced {combined.shape[1]} tokens for a sample of length {sample_len}."
        )
    return combined, {
        "image_index": image_index,
        "video_index": video_index,
        "ss_index": ss_index,
        "shape_index": shape_index,
        "tex_index": tex_index,
    }


def build_mot_position_ids(
    *,
    input_ids: torch.LongTensor,
    sample_lens: Optional[List[List[int]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    ss_grid_thw: Optional[Union[torch.Tensor, Sequence[Sequence[int]]]] = None,
    shape_slat: Optional[Any] = None,
    tex_slat: Optional[Any] = None,
    shape_coords: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    tex_coords: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    spatial_merge_size: int = 1,
    image_token_id: int = -1,
    video_token_id: int = -1,
    vision_start_token_id: int = -1,
    ss_start_token_id: Optional[int] = None,
    ss_pad_token_id: Optional[int] = None,
    ss_end_token_id: Optional[int] = None,
    shape_start_token_id: Optional[int] = None,
    shape_pad_token_id: Optional[int] = None,
    shape_end_token_id: Optional[int] = None,
    tex_start_token_id: Optional[int] = None,
    tex_pad_token_id: Optional[int] = None,
    tex_end_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build explicit MRoPE position IDs for packed or unpacked MoT sequences."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B, L], got {tuple(input_ids.shape)}.")

    batch_size, max_len = input_ids.shape
    device = input_ids.device
    position_ids = torch.zeros(3, batch_size, max_len, dtype=input_ids.dtype, device=device)

    ss_grid_list = _normalize_dense_grid_list(ss_grid_thw)
    shape_coord_blocks = (
        _extract_sparse_coords_from_runtime_tensor(shape_slat, name="shape_slat")
        if shape_slat is not None
        else _normalize_sparse_coord_list(shape_coords, name="shape_coords")
    )
    tex_coord_blocks = (
        _extract_sparse_coords_from_runtime_tensor(tex_slat, name="tex_slat")
        if tex_slat is not None
        else _normalize_sparse_coord_list(tex_coords, name="tex_coords")
    )

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    image_index = 0
    video_index = 0
    ss_index = 0
    shape_index = 0
    tex_index = 0

    if sample_lens is not None:
        for batch_index in range(batch_size):
            row_parts: List[torch.Tensor] = []
            row_cursor = 0
            for sample_len in sample_lens[batch_index]:
                sample_ids = input_ids[batch_index, row_cursor : row_cursor + sample_len]
                sample_positions, state = _compute_single_sample_mot_positions(
                    input_ids_sample=sample_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    ss_grid_thw=ss_grid_list,
                    shape_coord_blocks=shape_coord_blocks,
                    tex_coord_blocks=tex_coord_blocks,
                    image_index=image_index,
                    video_index=video_index,
                    ss_index=ss_index,
                    shape_index=shape_index,
                    tex_index=tex_index,
                    spatial_merge_size=spatial_merge_size,
                    image_token_id=image_token_id,
                    video_token_id=video_token_id,
                    vision_start_token_id=vision_start_token_id,
                    ss_start_token_id=ss_start_token_id,
                    ss_pad_token_id=ss_pad_token_id,
                    ss_end_token_id=ss_end_token_id,
                    shape_start_token_id=shape_start_token_id,
                    shape_pad_token_id=shape_pad_token_id,
                    shape_end_token_id=shape_end_token_id,
                    tex_start_token_id=tex_start_token_id,
                    tex_pad_token_id=tex_pad_token_id,
                    tex_end_token_id=tex_end_token_id,
                )
                row_parts.append(sample_positions)
                row_cursor += sample_len
                image_index = state["image_index"]
                video_index = state["video_index"]
                ss_index = state["ss_index"]
                shape_index = state["shape_index"]
                tex_index = state["tex_index"]
            combined = torch.cat(row_parts, dim=1) if row_parts else torch.empty((3, 0), device=device, dtype=torch.long)
            position_ids[:, batch_index, : combined.shape[1]] = combined
        deltas = compute_mrope_position_deltas(position_ids, max_len=max_len, sample_lens=sample_lens)
        return position_ids, deltas

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    for batch_index in range(batch_size):
        sample_ids = input_ids[batch_index][attention_mask[batch_index] == 1]
        sample_positions, state = _compute_single_sample_mot_positions(
            input_ids_sample=sample_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            ss_grid_thw=ss_grid_list,
            shape_coord_blocks=shape_coord_blocks,
            tex_coord_blocks=tex_coord_blocks,
            image_index=image_index,
            video_index=video_index,
            ss_index=ss_index,
            shape_index=shape_index,
            tex_index=tex_index,
            spatial_merge_size=spatial_merge_size,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            vision_start_token_id=vision_start_token_id,
            ss_start_token_id=ss_start_token_id,
            ss_pad_token_id=ss_pad_token_id,
            ss_end_token_id=ss_end_token_id,
            shape_start_token_id=shape_start_token_id,
            shape_pad_token_id=shape_pad_token_id,
            shape_end_token_id=shape_end_token_id,
            tex_start_token_id=tex_start_token_id,
            tex_pad_token_id=tex_pad_token_id,
            tex_end_token_id=tex_end_token_id,
        )
        position_ids[:, batch_index, attention_mask[batch_index] == 1] = sample_positions.to(position_ids.device)
        image_index = state["image_index"]
        video_index = state["video_index"]
        ss_index = state["ss_index"]
        shape_index = state["shape_index"]
        tex_index = state["tex_index"]

    deltas = compute_mrope_position_deltas(position_ids, max_len=max_len, attention_mask=attention_mask)
    return position_ids, deltas


def extend_rope_index_with_stage(
    input_ids: torch.LongTensor,
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    attention_mask: Optional[torch.Tensor],
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    ss_grid_thw: Optional[Union[torch.Tensor, Sequence[Sequence[int]]]] = None,
    shape_slat: Optional[Any] = None,
    tex_slat: Optional[Any] = None,
    shape_coords: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    tex_coords: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ss_start_token_id: Optional[int] = None,
    ss_pad_token_id: Optional[int] = None,
    ss_end_token_id: Optional[int] = None,
    shape_start_token_id: Optional[int] = None,
    shape_pad_token_id: Optional[int] = None,
    shape_end_token_id: Optional[int] = None,
    tex_start_token_id: Optional[int] = None,
    tex_pad_token_id: Optional[int] = None,
    tex_end_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build explicit MRoPE position IDs for a non-packed unified MoT row."""
    return build_mot_position_ids(
        input_ids=input_ids,
        sample_lens=None,
        attention_mask=attention_mask,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        ss_grid_thw=ss_grid_thw,
        shape_slat=shape_slat,
        tex_slat=tex_slat,
        shape_coords=shape_coords,
        tex_coords=tex_coords,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        ss_start_token_id=ss_start_token_id,
        ss_pad_token_id=ss_pad_token_id,
        ss_end_token_id=ss_end_token_id,
        shape_start_token_id=shape_start_token_id,
        shape_pad_token_id=shape_pad_token_id,
        shape_end_token_id=shape_end_token_id,
        tex_start_token_id=tex_start_token_id,
        tex_pad_token_id=tex_pad_token_id,
        tex_end_token_id=tex_end_token_id,
    )


def extend_rope_index_with_stage_packed(
    input_ids: torch.LongTensor,
    sample_lens: List[List[int]],
    image_grid_thw: Optional[torch.LongTensor],
    video_grid_thw: Optional[torch.LongTensor],
    spatial_merge_size: int,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
    ss_grid_thw: Optional[Union[torch.Tensor, Sequence[Sequence[int]]]] = None,
    shape_slat: Optional[Any] = None,
    tex_slat: Optional[Any] = None,
    shape_coords: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    tex_coords: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
    ss_start_token_id: Optional[int] = None,
    ss_pad_token_id: Optional[int] = None,
    ss_end_token_id: Optional[int] = None,
    shape_start_token_id: Optional[int] = None,
    shape_pad_token_id: Optional[int] = None,
    shape_end_token_id: Optional[int] = None,
    tex_start_token_id: Optional[int] = None,
    tex_pad_token_id: Optional[int] = None,
    tex_end_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build explicit MRoPE position IDs for packed unified MoT rows."""
    return build_mot_position_ids(
        input_ids=input_ids,
        sample_lens=sample_lens,
        attention_mask=None,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        ss_grid_thw=ss_grid_thw,
        shape_slat=shape_slat,
        tex_slat=tex_slat,
        shape_coords=shape_coords,
        tex_coords=tex_coords,
        spatial_merge_size=spatial_merge_size,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        vision_start_token_id=vision_start_token_id,
        ss_start_token_id=ss_start_token_id,
        ss_pad_token_id=ss_pad_token_id,
        ss_end_token_id=ss_end_token_id,
        shape_start_token_id=shape_start_token_id,
        shape_pad_token_id=shape_pad_token_id,
        shape_end_token_id=shape_end_token_id,
        tex_start_token_id=tex_start_token_id,
        tex_pad_token_id=tex_pad_token_id,
        tex_end_token_id=tex_end_token_id,
    )


def build_text_position_ids(
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build simple 1D position IDs for text tokens.

    For text-only sequences, position IDs are the same across all 3 MRoPE dimensions.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Position IDs of shape [3, seq_len]
    """
    pos = torch.arange(seq_len, device=device)
    return pos.unsqueeze(0).expand(3, -1)


def concat_position_ids(
    pos_ids_list: List[torch.Tensor],
) -> torch.Tensor:
    """Concatenate multiple position ID tensors.

    Args:
        pos_ids_list: List of position ID tensors, each of shape [3, ...] or [3, batch, ...]

    Returns:
        Concatenated position IDs
    """
    # Handle both batched and unbatched inputs
    if pos_ids_list[0].dim() == 2:
        # Unbatched: [3, len]
        return torch.cat(pos_ids_list, dim=1)
    else:
        # Batched: [3, batch, len]
        return torch.cat(pos_ids_list, dim=2)


def validate_position_ids(
    position_ids: torch.Tensor,
    max_position: int,
) -> bool:
    """Validate that position IDs don't exceed max position.

    This is important to avoid RoPE extrapolation issues.

    Args:
        position_ids: Position IDs tensor of shape [3, ...]
        max_position: Maximum allowed position

    Returns:
        True if all positions are valid
    """
    max_pos = position_ids.max().item()
    return max_pos <= max_position


__all__ = [
    "build_dense_raster_position_ids",
    "build_sparse_position_ids_from_coords",
    "build_mot_position_ids",
    "compute_mrope_position_deltas",
    "extend_rope_index_with_stage",
    "extend_rope_index_with_stage_packed",
    "build_text_position_ids",
    "concat_position_ids",
    "validate_position_ids",
]
