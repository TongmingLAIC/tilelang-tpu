# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""DeepSeek-V3 mixture-of-experts inference on BM1690.

Implements group-limited routing, routed and shared SwiGLU experts, and
weighted token aggregation, with a PyTorch reference for output validation.
"""

import os
from dataclasses import dataclass

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F


@dataclass
class DispatchMetadata:
    """Routing metadata used by the CPU reference implementation."""

    permute_indices: torch.Tensor
    scatter_indices: torch.Tensor
    expert_offsets: torch.Tensor
    topk_weights_flat: torch.Tensor
    expert_ids_flat: torch.Tensor
    topk_positions_flat: torch.Tensor


def reference_router_linear_deepseek(hidden_states, gate_weight):
    """Compute DeepSeek-V3 router logits with a linear projection."""

    return F.linear(hidden_states.to(torch.float32), gate_weight.to(torch.float32))


def build_dispatch_metadata(topk_indices, topk_weights, n_experts):
    """Convert per-token Top-K routing results to expert-major metadata."""

    if topk_indices.shape != topk_weights.shape:
        raise ValueError("topk_indices and topk_weights must have the same shape")

    topk_indices_cpu = topk_indices.to(torch.int64).cpu()
    topk_weights_cpu = topk_weights.to(torch.float32).cpu()
    n_tokens, top_k = topk_indices_cpu.shape

    permute_indices = []
    scatter_indices = []
    expert_ids_flat = []
    topk_positions_flat = []
    topk_weights_flat = []
    expert_offsets = [0]

    for expert_id in range(n_experts):
        for token_id in range(n_tokens):
            for topk_pos in range(top_k):
                if int(topk_indices_cpu[token_id, topk_pos]) == expert_id:
                    permute_indices.append(token_id)
                    scatter_indices.append(token_id)
                    expert_ids_flat.append(expert_id)
                    topk_positions_flat.append(topk_pos)
                    topk_weights_flat.append(float(topk_weights_cpu[token_id, topk_pos]))
        expert_offsets.append(len(permute_indices))

    return DispatchMetadata(
        permute_indices=torch.tensor(permute_indices, dtype=torch.int32),
        scatter_indices=torch.tensor(scatter_indices, dtype=torch.int32),
        expert_offsets=torch.tensor(expert_offsets, dtype=torch.int32),
        topk_weights_flat=torch.tensor(topk_weights_flat, dtype=torch.float32),
        expert_ids_flat=torch.tensor(expert_ids_flat, dtype=torch.int32),
        topk_positions_flat=torch.tensor(topk_positions_flat, dtype=torch.int32),
    )


def permute_hidden_cpu(hidden_states, permute_indices):
    """Permute hidden states into expert-major dispatch order."""

    return hidden_states[permute_indices.to(torch.int64)]


def combine_routed_outputs_cpu(routed_outputs, metadata, n_tokens):
    """Scale routed expert outputs and scatter-add them back to token order."""

    result = torch.zeros(n_tokens, routed_outputs.shape[1], dtype=routed_outputs.dtype)
    weights = metadata.topk_weights_flat.to(routed_outputs.dtype).unsqueeze(1)
    result.index_add_(
        0,
        metadata.scatter_indices.to(torch.int64),
        routed_outputs * weights,
    )
    return result


def moe_kernel(
    n_tokens,
    hidden_size,
    n_experts,
    n_group,
    topk_group,
    top_k,
    intermediate_size,
    n_shared_experts,
    block_m,
    block_h,
    block_e,
    block_i,
    routed_scaling_factor=1.0,
    dtype="float32",
    gemm_dtype="bfloat16",
):
    """Build a MoE kernel with DeepSeek-V3 routing semantics.

    Args:
        n_tokens: Number of input tokens.
        hidden_size: Hidden dimension of each token.
        n_experts: Number of routed experts.
        n_group: Number of expert groups used by group-limited routing.
        topk_group: Number of groups kept before the final expert Top-K.
        top_k: Number of routed experts selected for each token.
        intermediate_size: Per-expert SwiGLU intermediate dimension.
        n_shared_experts: Shared expert multiplier. The shared MLP uses
            intermediate_size * n_shared_experts hidden channels.
        block_m: Token tile size.
        block_h: Hidden tile size.
        block_e: Routed expert tile size for router linear.
        block_i: Intermediate tile size for expert MLP GEMMs.
        routed_scaling_factor: Scale applied after Top-K weight normalization.
        dtype: Storage dtype for inputs, outputs, and elementwise intermediates.
        gemm_dtype: GEMM input dtype.

    Routing selects expert groups and experts per token. Routed and shared
    expert outputs are combined into moe_output.
    """

    experts_per_group = n_experts // n_group
    shared_intermediate_size = intermediate_size * n_shared_experts
    group_rows = n_tokens * n_group
    selected_group_rows = n_tokens * topk_group
    final_index_rows = n_tokens * top_k
    token_blocks = n_tokens // block_m
    final_token_blocks = final_index_rows // block_m
    expert_blocks = n_experts // block_e
    hidden_blocks = hidden_size // block_h
    intermediate_blocks = intermediate_size // block_i
    shared_intermediate_blocks = shared_intermediate_size // block_i

    @T.macro
    def router_linear_stage(hidden_states, gate_weight, router_logits_workspace):
        """Compute router logits."""

        with T.Kernel(token_blocks, expert_blocks, is_cpu=True) as (bm, be):
            hidden_tile = T.alloc_shared((block_m, block_h), dtype)
            hidden_gemm = T.alloc_shared((block_m, block_h), gemm_dtype)
            weight_tile = T.alloc_shared((block_e, block_h), dtype)
            weight_gemm = T.alloc_shared((block_e, block_h), gemm_dtype)
            logits_acc = T.alloc_shared((block_m, block_e), dtype)
            logits_partial = T.alloc_shared((block_m, block_e), dtype)

            T.ppl_fill(logits_acc, T.float32(0.0))
            for hk in T.serial(hidden_blocks):
                T.ppl_copy(
                    hidden_states[
                        bm * block_m:(bm + 1) * block_m, hk * block_h:(hk + 1) * block_h
                    ],
                    hidden_tile,
                )
                T.ppl_copy(
                    gate_weight[
                        be * block_e:(be + 1) * block_e, hk * block_h:(hk + 1) * block_h
                    ],
                    weight_tile,
                )
                T.ppl_copy(hidden_tile, hidden_gemm)
                T.ppl_copy(weight_tile, weight_gemm)
                T.ppl_fill(logits_partial, T.float32(0.0))
                T.ppl_gemm(
                    hidden_gemm,
                    weight_gemm,
                    logits_partial,
                    transpose_B=True,
                    accumulate=False,
                )
                T.ppl_add(logits_acc, logits_acc, logits_partial)

            T.ppl_copy(
                logits_acc,
                router_logits_workspace[
                    bm * block_m:(bm + 1) * block_m, be * block_e:(be + 1) * block_e
                ],
            )

    @T.macro
    def router_postprocess_stage(
        router_logits_workspace,
        correction_bias,
        group_expert_mask_table,
        expert_onehot_table,
        selected_group_indices_workspace,
        selected_group_masks_workspace,
        final_expert_indices_workspace,
        selected_weights_by_flat_workspace,
        final_expert_onehot_masks_workspace,
    ):
        """Apply sigmoid routing and group-limited Top-K."""

        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            logits_tile = T.alloc_shared((n_tokens, n_experts), dtype)
            router_scores = T.alloc_shared((n_tokens, n_experts), dtype)
            bias_tile = T.alloc_shared((n_tokens, n_experts), dtype)
            choice_scores = T.alloc_shared((n_tokens, n_experts), dtype)
            sigmoid_work0 = T.alloc_shared((n_tokens, n_experts), dtype)
            sigmoid_work1 = T.alloc_shared((n_tokens, n_experts), dtype)
            sigmoid_coeff = T.alloc_shared((64, 32), dtype)

            group_choice_scores_tile = T.alloc_shared(
                (group_rows, experts_per_group), dtype
            )
            group_top2_scores_tile = T.alloc_shared(
                (group_rows, experts_per_group), dtype
            )
            group_top2_indices_tile = T.alloc_shared(
                (group_rows, experts_per_group), "int32"
            )
            group_top2_row = T.alloc_shared((1, 2), dtype)
            group_top2_score0 = T.alloc_shared((1, 1), dtype)
            group_top2_score1 = T.alloc_shared((1, 1), dtype)
            group_score_one = T.alloc_shared((1, 1), dtype)
            group_choice_one = T.alloc_shared((1, 1), dtype)
            group_scores_tile = T.alloc_shared((n_tokens, n_group), dtype)
            selected_group_scores_tile = T.alloc_shared((n_tokens, n_group), dtype)
            selected_group_indices_tile = T.alloc_shared(
                (n_tokens, n_group), "uint32"
            )
            final_topk_scores_tile = T.alloc_shared((n_tokens, n_experts), dtype)
            final_topk_indices_tile = T.alloc_shared(
                (n_tokens, n_experts), "uint32"
            )
            index_one = T.alloc_shared((1, 1), "uint32")
            selected_group_masks_tile = T.alloc_shared(
                (selected_group_rows, n_experts), dtype
            )
            final_expert_masks_tile = T.alloc_shared(
                (final_index_rows, n_experts), dtype
            )
            selected_weights_tile = T.alloc_shared(
                (final_index_rows, 1), dtype
            )

            group_mask_row = T.alloc_shared((1, n_experts), dtype)
            expert_mask_row = T.alloc_shared((1, n_experts), dtype)
            expert_mask_tile = T.alloc_shared((n_tokens, n_experts), dtype)
            ones_tile = T.alloc_shared((n_tokens, n_experts), dtype)
            inverse_mask_tile = T.alloc_shared((n_tokens, n_experts), dtype)
            kept_scores = T.alloc_shared((n_tokens, n_experts), dtype)
            masked_fill_scores = T.alloc_shared((n_tokens, n_experts), dtype)
            masked_scores = T.alloc_shared((n_tokens, n_experts), dtype)

            router_score_row = T.alloc_shared((1, n_experts), dtype)
            expert_onehot_row = T.alloc_shared((1, n_experts), dtype)
            selected_weight_row = T.alloc_shared((1, n_experts), dtype)
            selected_weight = T.alloc_shared((1, 1), dtype)
            weight_sum = T.alloc_shared((1, 1), dtype)
            normalized_weight = T.alloc_shared((1, 1), dtype)
            scaled_weight = T.alloc_shared((1, 1), dtype)

            T.ppl_copy(router_logits_workspace, logits_tile)
            T.ppl_copy(correction_bias, bias_tile)
            # sigmoid(x) = 1 / (1 + exp(-x)); preserve the input tile.
            T.ppl_mul_C(router_scores, logits_tile, T.float32(-1.0))
            T.ppl_exp(
                router_scores,
                sigmoid_work0,
                sigmoid_work1,
                sigmoid_coeff,
            )
            T.ppl_add_C(sigmoid_work1, router_scores, T.float32(1.0))
            T.ppl_fill(sigmoid_work0, T.float32(1.0))
            T.ppl_div(router_scores, sigmoid_work0, sigmoid_work1)
            T.ppl_add(choice_scores, router_scores, bias_tile)

            for token_id in T.serial(n_tokens):
                for group_id in T.serial(n_group):
                    for expert_in_group in T.serial(experts_per_group):
                        T.ppl_copy(
                            choice_scores[
                                token_id : token_id + 1,
                                group_id * experts_per_group
                                + expert_in_group : group_id * experts_per_group
                                + expert_in_group + 1,
                            ],
                            group_choice_one,
                        )
                        T.ppl_copy(
                            group_choice_one,
                            group_choice_scores_tile[
                                token_id * n_group + group_id : token_id * n_group
                                + group_id + 1,
                                expert_in_group : expert_in_group + 1,
                            ],
                        )

            T.ppl_topk(
                group_top2_scores_tile,
                group_top2_indices_tile,
                group_choice_scores_tile,
                2,
                True,
                experts_per_group,
            )

            for token_id in T.serial(n_tokens):
                for group_id in T.serial(n_group):
                    T.ppl_copy(
                        group_top2_scores_tile[
                            token_id * n_group + group_id:token_id * n_group + group_id + 1, 0:2
                        ],
                        group_top2_row,
                    )
                    T.ppl_copy(group_top2_row[0:1, 0:1], group_top2_score0)
                    T.ppl_copy(group_top2_row[0:1, 1:2], group_top2_score1)
                    T.ppl_add(group_score_one, group_top2_score0, group_top2_score1)
                    T.ppl_copy(
                        group_score_one,
                        group_scores_tile[
                            token_id:token_id + 1, group_id:group_id + 1
                        ],
                    )

            T.ppl_topk(
                selected_group_scores_tile,
                selected_group_indices_tile,
                group_scores_tile,
                topk_group,
                True,
                n_group,
            )

            for token_id in T.serial(n_tokens):
                for slot_id in T.serial(topk_group):
                    T.ppl_copy(
                        selected_group_indices_tile[
                            token_id:token_id + 1, slot_id:slot_id + 1
                        ],
                        index_one,
                    )
                    T.ppl_copy(
                        index_one,
                        selected_group_indices_workspace[
                            token_id * topk_group + slot_id:token_id * topk_group + slot_id + 1, 0:1
                        ],
                    )

            # Expand selected groups into expert masks.
            T.ppl_gather(
                selected_group_masks_workspace,
                group_expert_mask_table,
                selected_group_indices_workspace,
                n_group,
            )
            T.ppl_copy(selected_group_masks_workspace, selected_group_masks_tile)

            for token_id in T.serial(n_tokens):
                T.ppl_fill(expert_mask_row, T.float32(0))
                for slot_id in T.serial(topk_group):
                    T.ppl_copy(
                        selected_group_masks_tile[
                            token_id * topk_group + slot_id:token_id * topk_group + slot_id + 1, 0:n_experts
                        ],
                        group_mask_row,
                    )
                    T.ppl_add(expert_mask_row, expert_mask_row, group_mask_row)
                T.ppl_copy(
                    expert_mask_row,
                    expert_mask_tile[token_id:token_id + 1, 0:n_experts],
                )

            T.ppl_fill(ones_tile, T.float32(1))
            T.ppl_subtract(inverse_mask_tile, ones_tile, expert_mask_tile)
            T.ppl_mul(kept_scores, choice_scores, expert_mask_tile)
            T.ppl_mul_C(masked_fill_scores, inverse_mask_tile, T.float32(-1.0e9))
            T.ppl_add(masked_scores, kept_scores, masked_fill_scores)

            T.ppl_topk(
                final_topk_scores_tile,
                final_topk_indices_tile,
                masked_scores,
                top_k,
                True,
                n_experts,
            )

            for token_id in T.serial(n_tokens):
                for slot_id in T.serial(top_k):
                    T.ppl_copy(
                        final_topk_indices_tile[
                            token_id:token_id + 1, slot_id:slot_id + 1
                        ],
                        index_one,
                    )
                    T.ppl_copy(
                        index_one,
                        final_expert_indices_workspace[
                            token_id * top_k + slot_id:token_id * top_k + slot_id + 1, 0:1
                        ],
                    )

            # Build expert masks by selecting rows from the identity table.
            T.ppl_gather(
                final_expert_onehot_masks_workspace,
                expert_onehot_table,
                final_expert_indices_workspace,
                n_experts,
            )
            T.ppl_copy(final_expert_onehot_masks_workspace, final_expert_masks_tile)

            for token_id in T.serial(n_tokens):
                T.ppl_copy(
                    router_scores[token_id:token_id + 1, 0:n_experts],
                    router_score_row,
                )
                T.ppl_fill(weight_sum, T.float32(0))
                for slot_id in T.serial(top_k):
                    T.ppl_copy(
                        final_expert_masks_tile[
                            token_id * top_k + slot_id:token_id * top_k + slot_id + 1, 0:n_experts
                        ],
                        expert_onehot_row,
                    )
                    T.ppl_mul(selected_weight_row, router_score_row, expert_onehot_row)
                    T.ppl_reduce_sum(selected_weight_row, selected_weight, 1)
                    T.ppl_add(weight_sum, weight_sum, selected_weight)

                for slot_id in T.serial(top_k):
                    T.ppl_copy(
                        final_expert_masks_tile[
                            token_id * top_k + slot_id:token_id * top_k + slot_id + 1, 0:n_experts
                        ],
                        expert_onehot_row,
                    )
                    T.ppl_mul(selected_weight_row, router_score_row, expert_onehot_row)
                    T.ppl_reduce_sum(selected_weight_row, selected_weight, 1)
                    T.ppl_div(normalized_weight, selected_weight, weight_sum)
                    T.ppl_mul_C(
                        scaled_weight,
                        normalized_weight,
                        T.float32(float(routed_scaling_factor)),
                    )
                    T.ppl_copy(
                        scaled_weight,
                        selected_weights_tile[
                            token_id * top_k + slot_id:token_id * top_k + slot_id + 1, 0:1
                        ],
                    )

            T.ppl_copy(selected_weights_tile, selected_weights_by_flat_workspace)

    @T.macro
    def token_permutation_stage(
        hidden_states,
        expert_id_table,
        selection_order_scores,
        flat_to_token_index_table,
        selected_weights_by_flat_workspace,
        final_expert_onehot_masks_workspace,
        dispatch_sort_indices_workspace,
        permuted_token_indices_workspace,
        permuted_hidden_workspace,
        topk_weights_flat_workspace,
        permuted_expert_masks_workspace,
    ):
        """Build expert-major permutation metadata and permuted tensors."""

        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            expert_onehot_row = T.alloc_shared((1, n_experts), dtype)
            expert_id_row = T.alloc_shared((1, n_experts), dtype)
            expert_id_mul_row = T.alloc_shared((1, n_experts), dtype)
            expert_id_one = T.alloc_shared((1, 1), dtype)
            expert_key_one = T.alloc_shared((1, 1), dtype)
            order_score_one = T.alloc_shared((1, 1), dtype)
            positive_key_one = T.alloc_shared((1, 1), dtype)
            negative_key_one = T.alloc_shared((1, 1), dtype)
            index_one_u32 = T.alloc_shared((1, 1), "uint32")
            dispatch_sort_keys_tile = T.alloc_shared((1, final_index_rows), dtype)
            dispatch_sort_scores_tile = T.alloc_shared((1, final_index_rows), dtype)
            dispatch_sort_indices_tile = T.alloc_shared(
                (1, final_index_rows), "uint32"
            )

            for flat_id in T.serial(final_index_rows):
                T.ppl_copy(
                    final_expert_onehot_masks_workspace[
                        flat_id:flat_id + 1, 0:n_experts
                    ],
                    expert_onehot_row,
                )
                T.ppl_copy(expert_id_table[0:1, 0:n_experts], expert_id_row)
                T.ppl_mul(expert_id_mul_row, expert_onehot_row, expert_id_row)
                T.ppl_reduce_sum(expert_id_mul_row, expert_id_one, 1)
                T.ppl_mul_C(
                    expert_key_one,
                    expert_id_one,
                    T.float32(float(final_index_rows)),
                )
                T.ppl_copy(
                    selection_order_scores[flat_id:flat_id + 1, 0:1],
                    order_score_one,
                )
                T.ppl_add(positive_key_one, expert_key_one, order_score_one)
                T.ppl_mul_C(negative_key_one, positive_key_one, T.float32(-1.0))
                T.ppl_copy(
                    negative_key_one,
                    dispatch_sort_keys_tile[0:1, flat_id:flat_id + 1],
                )

            T.ppl_topk(
                dispatch_sort_scores_tile,
                dispatch_sort_indices_tile,
                dispatch_sort_keys_tile,
                final_index_rows,
                True,
                final_index_rows,
            )
            for flat_id in T.serial(final_index_rows):
                T.ppl_copy(
                    dispatch_sort_indices_tile[0:1, flat_id:flat_id + 1],
                    index_one_u32,
                )
                T.ppl_copy(
                    index_one_u32,
                    dispatch_sort_indices_workspace[flat_id:flat_id + 1, 0:1],
                )
            # Reorder token indices, hidden states, routing weights, and expert masks.
            T.ppl_gather(
                permuted_token_indices_workspace,
                flat_to_token_index_table,
                dispatch_sort_indices_workspace,
                final_index_rows,
            )
            T.ppl_gather(
                permuted_hidden_workspace,
                hidden_states,
                permuted_token_indices_workspace,
                n_tokens,
            )
            T.ppl_gather(
                topk_weights_flat_workspace,
                selected_weights_by_flat_workspace,
                dispatch_sort_indices_workspace,
                final_index_rows,
            )
            T.ppl_gather(
                permuted_expert_masks_workspace,
                final_expert_onehot_masks_workspace,
                dispatch_sort_indices_workspace,
                final_index_rows,
            )

    @T.macro
    def routed_expert_stage(
        permuted_hidden_workspace,
        topk_weights_flat_workspace,
        permuted_expert_masks_workspace,
        gate_up_proj,
        down_proj,
        weighted_expert_output_workspace,
    ):
        """Compute routed expert MLP output for each row/hidden tile."""

        with T.Kernel(final_token_blocks, hidden_blocks, is_cpu=True) as (
            row_block,
            bh,
        ):
            expert_tokens = T.alloc_shared((block_m, block_h), dtype)
            expert_tokens_gemm = T.alloc_shared((block_m, block_h), gemm_dtype)
            gate_weight_tile = T.alloc_shared((block_i, block_h), dtype)
            gate_weight_gemm = T.alloc_shared((block_i, block_h), gemm_dtype)
            up_weight_tile = T.alloc_shared((block_i, block_h), dtype)
            up_weight_gemm = T.alloc_shared((block_i, block_h), gemm_dtype)
            down_weight_tile = T.alloc_shared((block_h, block_i), dtype)
            down_weight_gemm = T.alloc_shared((block_h, block_i), gemm_dtype)
            gate_tile = T.alloc_shared((block_m, block_i), dtype)
            gate_partial = T.alloc_shared((block_m, block_i), dtype)
            up_tile = T.alloc_shared((block_m, block_i), dtype)
            up_partial = T.alloc_shared((block_m, block_i), dtype)
            expert_sigmoid = T.alloc_shared((block_m, block_i), dtype)
            silu_gate = T.alloc_shared((block_m, block_i), dtype)
            activated = T.alloc_shared((block_m, block_i), dtype)
            activated_gemm = T.alloc_shared((block_m, block_i), gemm_dtype)
            result_acc = T.alloc_shared((block_m, block_h), dtype)
            result_partial = T.alloc_shared((block_m, block_h), dtype)
            expert_acc = T.alloc_shared((block_m, block_h), dtype)
            expert_selected = T.alloc_shared((block_m, block_h), dtype)
            expert_select_mask = T.alloc_shared((block_m, 1), dtype)
            expert_select_mask_full = T.alloc_shared((block_m, block_h), dtype)
            route_weight_tile = T.alloc_shared((block_m, 1), dtype)
            route_weight_full = T.alloc_shared((block_m, block_h), dtype)
            weighted_tile = T.alloc_shared((block_m, block_h), dtype)
            expert_sigmoid_work0 = T.alloc_shared((block_m, block_i), dtype)
            expert_sigmoid_work1 = T.alloc_shared((block_m, block_i), dtype)
            expert_sigmoid_coeff = T.alloc_shared((64, 32), dtype)

            T.ppl_fill(expert_acc, T.float32(0.0))

            for expert_id in T.serial(n_experts):
                T.ppl_fill(result_acc, T.float32(0.0))

                for bi in T.serial(intermediate_blocks):
                    T.ppl_fill(gate_tile, T.float32(0.0))
                    T.ppl_fill(up_tile, T.float32(0.0))

                    for hk in T.serial(hidden_blocks):
                        T.ppl_copy(
                            permuted_hidden_workspace[
                                row_block * block_m : (row_block + 1) * block_m,
                                hk * block_h : (hk + 1) * block_h,
                            ],
                            expert_tokens,
                        )
                        T.ppl_copy(
                            gate_up_proj[
                                expert_id * 2 * intermediate_size
                                + bi * block_i : expert_id * 2 * intermediate_size
                                + (bi + 1) * block_i,
                                hk * block_h : (hk + 1) * block_h,
                            ],
                            gate_weight_tile,
                        )
                        T.ppl_copy(
                            gate_up_proj[
                                expert_id * 2 * intermediate_size
                                + intermediate_size
                                + bi * block_i : expert_id * 2 * intermediate_size
                                + intermediate_size + (bi + 1) * block_i,
                                hk * block_h : (hk + 1) * block_h,
                            ],
                            up_weight_tile,
                        )
                        T.ppl_copy(expert_tokens, expert_tokens_gemm)
                        T.ppl_copy(gate_weight_tile, gate_weight_gemm)
                        T.ppl_copy(up_weight_tile, up_weight_gemm)

                        T.ppl_fill(gate_partial, T.float32(0.0))
                        T.ppl_gemm(
                            expert_tokens_gemm,
                            gate_weight_gemm,
                            gate_partial,
                            transpose_B=True,
                            accumulate=False,
                        )
                        T.ppl_add(gate_tile, gate_tile, gate_partial)

                        T.ppl_fill(up_partial, T.float32(0.0))
                        T.ppl_gemm(
                            expert_tokens_gemm,
                            up_weight_gemm,
                            up_partial,
                            transpose_B=True,
                            accumulate=False,
                        )
                        T.ppl_add(up_tile, up_tile, up_partial)

                    T.ppl_mul_C(expert_sigmoid, gate_tile, T.float32(-1.0))
                    T.ppl_exp(
                        expert_sigmoid,
                        expert_sigmoid_work0,
                        expert_sigmoid_work1,
                        expert_sigmoid_coeff,
                    )
                    T.ppl_add_C(expert_sigmoid_work1, expert_sigmoid, T.float32(1.0))
                    T.ppl_fill(expert_sigmoid_work0, T.float32(1.0))
                    T.ppl_div(expert_sigmoid, expert_sigmoid_work0, expert_sigmoid_work1)
                    T.ppl_mul(silu_gate, gate_tile, expert_sigmoid)
                    T.ppl_mul(activated, silu_gate, up_tile)
                    T.ppl_copy(activated, activated_gemm)
                    T.ppl_copy(
                        down_proj[
                            expert_id * hidden_size
                            + bh * block_h : expert_id * hidden_size
                            + (bh + 1) * block_h,
                            bi * block_i : (bi + 1) * block_i,
                        ],
                        down_weight_tile,
                    )
                    T.ppl_copy(down_weight_tile, down_weight_gemm)
                    T.ppl_fill(result_partial, T.float32(0.0))
                    T.ppl_gemm(
                        activated_gemm,
                        down_weight_gemm,
                        result_partial,
                        transpose_B=True,
                        accumulate=False,
                    )
                    T.ppl_add(result_acc, result_acc, result_partial)

                T.ppl_copy(
                    permuted_expert_masks_workspace[
                        row_block * block_m:(row_block + 1) * block_m, expert_id:expert_id + 1
                    ],
                    expert_select_mask,
                )
                for hidden_col in T.serial(block_h):
                    T.ppl_copy(
                        expert_select_mask,
                        expert_select_mask_full[
                            0:block_m, hidden_col:hidden_col + 1
                        ],
                    )
                T.ppl_mul(expert_selected, result_acc, expert_select_mask_full)
                T.ppl_add(expert_acc, expert_acc, expert_selected)

            T.ppl_copy(
                topk_weights_flat_workspace[
                    row_block * block_m:(row_block + 1) * block_m, 0:1
                ],
                route_weight_tile,
            )
            for hidden_col in T.serial(block_h):
                T.ppl_copy(
                    route_weight_tile,
                    route_weight_full[
                        0:block_m, hidden_col:hidden_col + 1
                    ],
                )
            T.ppl_mul(weighted_tile, expert_acc, route_weight_full)
            T.ppl_copy(
                weighted_tile,
                weighted_expert_output_workspace[
                    row_block * block_m:(row_block + 1) * block_m, bh * block_h:(bh + 1) * block_h
                ],
            )

    @T.macro
    def shared_expert_stage(
        hidden_states,
        shared_gate_up_proj,
        shared_down_proj,
        shared_output_workspace,
    ):
        """Compute shared expert MLP output for each token/hidden tile."""

        with T.Kernel(token_blocks, hidden_blocks, is_cpu=True) as (bm, bh):
            expert_tokens = T.alloc_shared((block_m, block_h), dtype)
            expert_tokens_gemm = T.alloc_shared((block_m, block_h), gemm_dtype)
            gate_weight_tile = T.alloc_shared((block_i, block_h), dtype)
            gate_weight_gemm = T.alloc_shared((block_i, block_h), gemm_dtype)
            up_weight_tile = T.alloc_shared((block_i, block_h), dtype)
            up_weight_gemm = T.alloc_shared((block_i, block_h), gemm_dtype)
            down_weight_tile = T.alloc_shared((block_h, block_i), dtype)
            down_weight_gemm = T.alloc_shared((block_h, block_i), gemm_dtype)
            gate_tile = T.alloc_shared((block_m, block_i), dtype)
            gate_partial = T.alloc_shared((block_m, block_i), dtype)
            up_tile = T.alloc_shared((block_m, block_i), dtype)
            up_partial = T.alloc_shared((block_m, block_i), dtype)
            expert_sigmoid = T.alloc_shared((block_m, block_i), dtype)
            silu_gate = T.alloc_shared((block_m, block_i), dtype)
            activated = T.alloc_shared((block_m, block_i), dtype)
            activated_gemm = T.alloc_shared((block_m, block_i), gemm_dtype)
            result_acc = T.alloc_shared((block_m, block_h), dtype)
            result_partial = T.alloc_shared((block_m, block_h), dtype)
            expert_sigmoid_work0 = T.alloc_shared((block_m, block_i), dtype)
            expert_sigmoid_work1 = T.alloc_shared((block_m, block_i), dtype)
            expert_sigmoid_coeff = T.alloc_shared((64, 32), dtype)

            T.ppl_fill(result_acc, T.float32(0.0))

            for bi in T.serial(shared_intermediate_blocks):
                T.ppl_fill(gate_tile, T.float32(0.0))
                T.ppl_fill(up_tile, T.float32(0.0))

                for hk in T.serial(hidden_blocks):
                    T.ppl_copy(
                        hidden_states[
                            bm * block_m:(bm + 1) * block_m, hk * block_h:(hk + 1) * block_h
                        ],
                        expert_tokens,
                    )
                    T.ppl_copy(
                        shared_gate_up_proj[
                            bi * block_i:(bi + 1) * block_i, hk * block_h:(hk + 1) * block_h
                        ],
                        gate_weight_tile,
                    )
                    T.ppl_copy(
                        shared_gate_up_proj[
                            shared_intermediate_size
                            + bi * block_i : shared_intermediate_size
                            + (bi + 1) * block_i,
                            hk * block_h : (hk + 1) * block_h,
                        ],
                        up_weight_tile,
                    )
                    T.ppl_copy(expert_tokens, expert_tokens_gemm)
                    T.ppl_copy(gate_weight_tile, gate_weight_gemm)
                    T.ppl_copy(up_weight_tile, up_weight_gemm)

                    T.ppl_fill(gate_partial, T.float32(0.0))
                    T.ppl_gemm(
                        expert_tokens_gemm,
                        gate_weight_gemm,
                        gate_partial,
                        transpose_B=True,
                        accumulate=False,
                    )
                    T.ppl_add(gate_tile, gate_tile, gate_partial)

                    T.ppl_fill(up_partial, T.float32(0.0))
                    T.ppl_gemm(
                        expert_tokens_gemm,
                        up_weight_gemm,
                        up_partial,
                        transpose_B=True,
                        accumulate=False,
                    )
                    T.ppl_add(up_tile, up_tile, up_partial)

                T.ppl_mul_C(expert_sigmoid, gate_tile, T.float32(-1.0))
                T.ppl_exp(
                    expert_sigmoid,
                    expert_sigmoid_work0,
                    expert_sigmoid_work1,
                    expert_sigmoid_coeff,
                )
                T.ppl_add_C(expert_sigmoid_work1, expert_sigmoid, T.float32(1.0))
                T.ppl_fill(expert_sigmoid_work0, T.float32(1.0))
                T.ppl_div(expert_sigmoid, expert_sigmoid_work0, expert_sigmoid_work1)
                T.ppl_mul(silu_gate, gate_tile, expert_sigmoid)
                T.ppl_mul(activated, silu_gate, up_tile)
                T.ppl_copy(activated, activated_gemm)
                T.ppl_copy(
                    shared_down_proj[
                        bh * block_h:(bh + 1) * block_h, bi * block_i:(bi + 1) * block_i
                    ],
                    down_weight_tile,
                )
                T.ppl_copy(down_weight_tile, down_weight_gemm)
                T.ppl_fill(result_partial, T.float32(0.0))
                T.ppl_gemm(
                    activated_gemm,
                    down_weight_gemm,
                    result_partial,
                    transpose_B=True,
                    accumulate=False,
                )
                T.ppl_add(result_acc, result_acc, result_partial)

            T.ppl_copy(
                result_acc,
                shared_output_workspace[
                    bm * block_m:(bm + 1) * block_m, bh * block_h:(bh + 1) * block_h
                ],
            )

    @T.macro
    def final_scatter_stage(
        shared_output_workspace,
        weighted_expert_output_workspace,
        permuted_token_indices_workspace,
        moe_output,
    ):
        """Combine shared and weighted routed expert outputs."""

        with T.Kernel(1, 1, is_cpu=True) as (_, __):
            initial_tile = T.alloc_shared((block_m, block_h), dtype)
            # Start with shared expert outputs, then add routed contributions.
            for bm in T.serial(token_blocks):
                for bh in T.serial(hidden_blocks):
                    T.ppl_copy(
                        shared_output_workspace[
                            bm * block_m : (bm + 1) * block_m,
                            bh * block_h : (bh + 1) * block_h,
                        ],
                        initial_tile,
                    )
                    T.ppl_copy(
                        initial_tile,
                        moe_output[
                            bm * block_m : (bm + 1) * block_m,
                            bh * block_h : (bh + 1) * block_h,
                        ],
                    )
            with T.block("scatter_weighted_experts_to_tokens"):
                T.reads(
                    weighted_expert_output_workspace[
                        0:final_index_rows, 0:hidden_size
                    ],
                    permuted_token_indices_workspace[0:final_index_rows, 0:1],
                    moe_output[0:n_tokens, 0:hidden_size],
                )
                T.writes(moe_output[0:n_tokens, 0:hidden_size])
                T.ppl_scatter_add(
                    moe_output,
                    weighted_expert_output_workspace,
                    permuted_token_indices_workspace,
                    0,
                )

    # Model tensors, routing metadata, workspaces, and the output buffer.
    @T.prim_func
    def main_kernel_inner(
        hidden_states: T.Tensor((n_tokens, hidden_size), dtype),
        gate_weight: T.Tensor((n_experts, hidden_size), dtype),
        correction_bias: T.Tensor((n_tokens, n_experts), dtype),
        group_expert_mask_table: T.Tensor((n_group, n_experts), dtype),
        expert_id_table: T.Tensor((1, n_experts), dtype),
        selection_order_scores: T.Tensor((final_index_rows, 1), dtype),
        gate_up_proj: T.Tensor(
            (n_experts * 2 * intermediate_size, hidden_size), dtype
        ),
        down_proj: T.Tensor((n_experts * hidden_size, intermediate_size), dtype),
        shared_gate_up_proj: T.Tensor(
            (2 * shared_intermediate_size, hidden_size), dtype
        ),
        shared_down_proj: T.Tensor((hidden_size, shared_intermediate_size), dtype),
        flat_to_token_index_table: T.Tensor((final_index_rows, 1), "uint32"),
        expert_onehot_table: T.Tensor((n_experts, n_experts), dtype),
        selected_group_indices_workspace: T.Tensor((selected_group_rows, 1), "uint32"),
        selected_group_masks_workspace: T.Tensor((selected_group_rows, n_experts), dtype),
        final_expert_indices_workspace: T.Tensor((final_index_rows, 1), "uint32"),
        dispatch_sort_indices_workspace: T.Tensor((final_index_rows, 1), "uint32"),
        router_logits_workspace: T.Tensor((n_tokens, n_experts), dtype),
        router_selected_weights_by_flat_workspace: T.Tensor(
            (final_index_rows, 1), dtype
        ),
        router_final_expert_onehot_masks_workspace: T.Tensor(
            (final_index_rows, n_experts), dtype
        ),
        permuted_token_indices_workspace: T.Tensor(
            (final_index_rows, 1), "uint32"
        ),
        permuted_hidden_workspace: T.Tensor((final_index_rows, hidden_size), dtype),
        topk_weights_flat_workspace: T.Tensor((final_index_rows, 1), dtype),
        permuted_expert_masks_workspace: T.Tensor(
            (final_index_rows, n_experts), dtype
        ),
        weighted_expert_output_workspace: T.Tensor(
            (final_index_rows, hidden_size), dtype
        ),
        shared_output_workspace: T.Tensor((n_tokens, hidden_size), dtype),
        moe_output: T.Tensor((n_tokens, hidden_size), dtype),
    ):
        router_linear_stage(hidden_states, gate_weight, router_logits_workspace)
        router_postprocess_stage(
            router_logits_workspace,
            correction_bias,
            group_expert_mask_table,
            expert_onehot_table,
            selected_group_indices_workspace,
            selected_group_masks_workspace,
            final_expert_indices_workspace,
            router_selected_weights_by_flat_workspace,
            router_final_expert_onehot_masks_workspace,
        )
        token_permutation_stage(
            hidden_states,
            expert_id_table,
            selection_order_scores,
            flat_to_token_index_table,
            router_selected_weights_by_flat_workspace,
            router_final_expert_onehot_masks_workspace,
            dispatch_sort_indices_workspace,
            permuted_token_indices_workspace,
            permuted_hidden_workspace,
            topk_weights_flat_workspace,
            permuted_expert_masks_workspace,
        )
        routed_expert_stage(
            permuted_hidden_workspace,
            topk_weights_flat_workspace,
            permuted_expert_masks_workspace,
            gate_up_proj,
            down_proj,
            weighted_expert_output_workspace,
        )
        shared_expert_stage(
            hidden_states,
            shared_gate_up_proj,
            shared_down_proj,
            shared_output_workspace,
        )
        final_scatter_stage(
            shared_output_workspace,
            weighted_expert_output_workspace,
            permuted_token_indices_workspace,
            moe_output,
        )

    return main_kernel_inner


def reference_multi_expert_deepseek(
    tokens,
    expert_ids_flat,
    gate_up_proj,
    down_proj,
    hidden_size,
    intermediate_size,
):
    """Compute routed expert MLP outputs in expert-major order."""

    n_tokens = tokens.shape[0]
    n_experts = gate_up_proj.shape[0] // (2 * intermediate_size)
    output = torch.zeros(n_tokens, hidden_size, dtype=torch.float32)
    expert_ids_flat = expert_ids_flat.to(torch.int64)

    for expert_id in range(n_experts):
        token_mask = expert_ids_flat == expert_id
        if not token_mask.any():
            continue

        expert_tokens = tokens[token_mask].to(torch.float32)
        base = expert_id * 2 * intermediate_size
        gate_weight = gate_up_proj[base : base + intermediate_size].to(torch.float32)
        up_weight = gate_up_proj[
            base + intermediate_size : base + 2 * intermediate_size
        ].to(torch.float32)
        down_weight = down_proj[
            expert_id * hidden_size : (expert_id + 1) * hidden_size
        ].to(torch.float32)

        gate = F.linear(expert_tokens, gate_weight)
        up = F.linear(expert_tokens, up_weight)
        output[token_mask] = F.linear(F.silu(gate) * up, down_weight)

    return output


def reference_shared_expert_deepseek(
    hidden_states,
    shared_gate_up_proj,
    shared_down_proj,
):
    """Compute the DeepSeek-V3 shared expert SwiGLU MLP."""

    shared_intermediate_size = shared_down_proj.shape[1]
    hidden_states = hidden_states.to(torch.float32)
    gate_weight = shared_gate_up_proj[:shared_intermediate_size].to(torch.float32)
    up_weight = shared_gate_up_proj[shared_intermediate_size:].to(torch.float32)
    down_weight = shared_down_proj.to(torch.float32)

    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)


def env_int(name, default):
    """Read an integer environment override."""

    return int(os.environ.get(name, default))


def env_float(name, default):
    """Read a floating-point environment override."""

    return float(os.environ.get(name, default))


def check_tensor(name, actual, expected, *, exact=False, rtol=5e-2, atol=1e-3):
    """Compare the final output with an independent PyTorch reference."""

    actual = actual.to(expected.dtype)
    if exact:
        matched = torch.equal(actual, expected)
        print(f"{name} exact match: {matched}")
    else:
        difference = (actual - expected).abs()
        matched = torch.allclose(actual, expected, rtol=rtol, atol=atol)
        print(f"{name} max absolute difference: {difference.max().item():.6e}")
        print(f"{name} match (rtol={rtol:g}, atol={atol:g}): {matched}")
    return matched


# Model dimensions and routing parameters.
N_TOKENS = env_int("TL_MOE_N_TOKENS", 8)
HIDDEN_SIZE = env_int("TL_MOE_HIDDEN_SIZE", 16)
N_EXPERTS = env_int("TL_MOE_N_EXPERTS", 8)
N_GROUP = env_int("TL_MOE_N_GROUP", 4)
TOPK_GROUP = env_int("TL_MOE_TOPK_GROUP", 2)
TOP_K = env_int("TL_MOE_TOP_K", 2)
INTERMEDIATE_SIZE = env_int("TL_MOE_INTERMEDIATE_SIZE", 16)
N_SHARED_EXPERTS = env_int("TL_MOE_N_SHARED_EXPERTS", 1)

# Tile sizes used by the kernel.
BLOCK_M = env_int("TL_MOE_BLOCK_M", 4)
BLOCK_H = env_int("TL_MOE_BLOCK_H", 8)
BLOCK_E = env_int("TL_MOE_BLOCK_E", 4)
BLOCK_I = env_int("TL_MOE_BLOCK_I", 8)
ROUTED_SCALING_FACTOR = env_float("TL_MOE_ROUTED_SCALING_FACTOR", 2.5)

assert N_TOKENS % BLOCK_M == 0, "N_TOKENS must be divisible by BLOCK_M"
assert (N_TOKENS * TOP_K) % BLOCK_M == 0, "N_TOKENS * TOP_K must be divisible by BLOCK_M"
assert HIDDEN_SIZE % BLOCK_H == 0, "HIDDEN_SIZE must be divisible by BLOCK_H"
assert N_EXPERTS % N_GROUP == 0, "N_EXPERTS must be divisible by N_GROUP"
assert N_EXPERTS % BLOCK_E == 0, "N_EXPERTS must be divisible by BLOCK_E"
assert INTERMEDIATE_SIZE % BLOCK_I == 0, "INTERMEDIATE_SIZE must be divisible by BLOCK_I"
assert N_SHARED_EXPERTS >= 1, "N_SHARED_EXPERTS must be >= 1"
assert (INTERMEDIATE_SIZE * N_SHARED_EXPERTS) % BLOCK_I == 0, (
    "INTERMEDIATE_SIZE * N_SHARED_EXPERTS must be divisible by BLOCK_I"
)
assert TOPK_GROUP <= N_GROUP, "TOPK_GROUP must be <= N_GROUP"
assert TOP_K >= 2, "TOP_K must be >= 2"
assert TOP_K <= TOPK_GROUP * (N_EXPERTS // N_GROUP), "TOP_K exceeds selected expert capacity"
assert N_EXPERTS // N_GROUP >= 2, "Example inputs require >= 2 experts per group"
assert N_TOKENS <= HIDDEN_SIZE, "Example inputs require N_TOKENS <= HIDDEN_SIZE"
EXPERTS_PER_GROUP = N_EXPERTS // N_GROUP
SHARED_INTERMEDIATE_SIZE = INTERMEDIATE_SIZE * N_SHARED_EXPERTS
GROUP_ROWS = N_TOKENS * N_GROUP
SELECTED_GROUP_ROWS = N_TOKENS * TOPK_GROUP
FINAL_INDEX_ROWS = N_TOKENS * TOP_K
NEG_MASK_VALUE = -1.0e9
MODE = os.environ.get("TL_MOE_MODE", "cmodel")


kernel = tilelang.compile(
    moe_kernel(
        N_TOKENS,
        HIDDEN_SIZE,
        N_EXPERTS,
        N_GROUP,
        TOPK_GROUP,
        TOP_K,
        INTERMEDIATE_SIZE,
        N_SHARED_EXPERTS,
        BLOCK_M,
        BLOCK_H,
        BLOCK_E,
        BLOCK_I,
        ROUTED_SCALING_FACTOR,
    ),
    out_idx=[25],
    target="tpu -mcpu=bm1690 -tpu-programming-model=tpukernel",
    runtime_mode=MODE,
)

SEED = env_int("TL_MOE_SEED", 73)
torch.manual_seed(SEED)

# Example inputs give every token a deterministic winning expert group.
hidden_states = torch.zeros(N_TOKENS, HIDDEN_SIZE, dtype=torch.float32)
gate_weight = torch.randn(N_EXPERTS, HIDDEN_SIZE, dtype=torch.float32) * 0.02
expert_correction_bias = torch.linspace(-0.03, 0.03, N_EXPERTS, dtype=torch.float32)
correction_bias = (
    expert_correction_bias.reshape(1, N_EXPERTS).expand(N_TOKENS, -1).contiguous()
)

forced_groups = torch.arange(N_TOKENS, dtype=torch.int64) % N_GROUP
for token_id, group_id in enumerate(forced_groups.tolist()):
    feature_id = token_id % HIDDEN_SIZE
    hidden_states[token_id, feature_id] = 1.0
    start = group_id * EXPERTS_PER_GROUP
    gate_weight[start, feature_id] = 3.0 + token_id * 0.01
    gate_weight[start + 1, feature_id] = 2.9 + token_id * 0.01

gate_up_proj = torch.randn(
    N_EXPERTS * 2 * INTERMEDIATE_SIZE,
    HIDDEN_SIZE,
    dtype=torch.float32,
) * 0.18
down_proj = torch.randn(
    N_EXPERTS * HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    dtype=torch.float32,
) * 0.18
shared_gate_up_proj = torch.randn(
    2 * SHARED_INTERMEDIATE_SIZE,
    HIDDEN_SIZE,
    dtype=torch.float32,
) * 0.18
shared_down_proj = torch.randn(
    HIDDEN_SIZE,
    SHARED_INTERMEDIATE_SIZE,
    dtype=torch.float32,
) * 0.18

group_expert_mask_table = torch.zeros(N_GROUP, N_EXPERTS, dtype=torch.float32)
for group_id in range(N_GROUP):
    start = group_id * EXPERTS_PER_GROUP
    group_expert_mask_table[group_id, start : start + EXPERTS_PER_GROUP] = 1.0

# Compact helper tables used by routing metadata generation.
expert_id_table = torch.arange(N_EXPERTS, dtype=torch.float32).reshape(1, N_EXPERTS)
selection_order_scores = torch.arange(FINAL_INDEX_ROWS, dtype=torch.float32).reshape(
    FINAL_INDEX_ROWS,
    1,
)

flat_to_token_index_table = (
    (torch.arange(FINAL_INDEX_ROWS, dtype=torch.int32) // TOP_K)
    .reshape(FINAL_INDEX_ROWS, 1)
    .contiguous()
).view(torch.uint32)

# Allocate routing and expert workspaces.
expert_onehot_table = torch.eye(N_EXPERTS, dtype=torch.float32)
selected_group_indices_workspace = torch.empty(SELECTED_GROUP_ROWS, 1, dtype=torch.uint32)
selected_group_masks_workspace = torch.empty(SELECTED_GROUP_ROWS, N_EXPERTS, dtype=torch.float32)
final_expert_indices_workspace = torch.empty(FINAL_INDEX_ROWS, 1, dtype=torch.uint32)
dispatch_sort_indices_workspace = torch.empty(FINAL_INDEX_ROWS, 1, dtype=torch.uint32)
router_logits_workspace = torch.empty(N_TOKENS, N_EXPERTS, dtype=torch.float32)
router_selected_weights_by_flat_workspace = torch.empty(FINAL_INDEX_ROWS, 1, dtype=torch.float32)
router_final_expert_onehot_masks_workspace = torch.empty(
    FINAL_INDEX_ROWS, N_EXPERTS, dtype=torch.float32
)
permuted_token_indices_workspace = torch.empty(FINAL_INDEX_ROWS, 1, dtype=torch.uint32)
permuted_hidden_workspace = torch.empty(FINAL_INDEX_ROWS, HIDDEN_SIZE, dtype=torch.float32)
topk_weights_flat_workspace = torch.empty(FINAL_INDEX_ROWS, 1, dtype=torch.float32)
permuted_expert_masks_workspace = torch.empty(FINAL_INDEX_ROWS, N_EXPERTS, dtype=torch.float32)
weighted_expert_output_workspace = torch.empty(FINAL_INDEX_ROWS, HIDDEN_SIZE, dtype=torch.float32)
shared_output_workspace = torch.empty(N_TOKENS, HIDDEN_SIZE, dtype=torch.float32)
moe_output = torch.empty(N_TOKENS, HIDDEN_SIZE, dtype=torch.float32)
assert flat_to_token_index_table.dtype == torch.uint32

kernel(
    hidden_states,
    gate_weight,
    correction_bias,
    group_expert_mask_table,
    expert_id_table,
    selection_order_scores,
    gate_up_proj,
    down_proj,
    shared_gate_up_proj,
    shared_down_proj,
    flat_to_token_index_table,
    expert_onehot_table,
    selected_group_indices_workspace,
    selected_group_masks_workspace,
    final_expert_indices_workspace,
    dispatch_sort_indices_workspace,
    router_logits_workspace,
    router_selected_weights_by_flat_workspace,
    router_final_expert_onehot_masks_workspace,
    permuted_token_indices_workspace,
    permuted_hidden_workspace,
    topk_weights_flat_workspace,
    permuted_expert_masks_workspace,
    weighted_expert_output_workspace,
    shared_output_workspace,
    moe_output,
)

router_logits_ref = reference_router_linear_deepseek(hidden_states, gate_weight)
router_scores_ref = torch.sigmoid(router_logits_ref)
choice_scores_ref = router_scores_ref + correction_bias
grouped_scores_ref = choice_scores_ref.reshape(N_TOKENS, N_GROUP, EXPERTS_PER_GROUP)
group_top2_ref, _ = torch.topk(grouped_scores_ref, 2, dim=-1, largest=True, sorted=False)
group_scores_ref = group_top2_ref.sum(dim=-1)
_, selected_group_indices_ref = torch.topk(
    group_scores_ref,
    TOPK_GROUP,
    dim=1,
    largest=True,
    sorted=True,
)
group_mask_ref = torch.zeros(N_TOKENS, N_GROUP, dtype=torch.float32)
group_mask_ref.scatter_(1, selected_group_indices_ref, 1.0)
expert_mask_ref = (
    group_mask_ref.unsqueeze(-1)
    .expand(N_TOKENS, N_GROUP, EXPERTS_PER_GROUP)
    .reshape(N_TOKENS, N_EXPERTS)
    .contiguous()
)
masked_scores_ref = (
    choice_scores_ref * expert_mask_ref + NEG_MASK_VALUE * (1.0 - expert_mask_ref)
)
_, final_topk_indices_ref = torch.topk(
    masked_scores_ref,
    TOP_K,
    dim=1,
    largest=True,
    sorted=True,
)
topk_weights_raw_ref = router_scores_ref.gather(1, final_topk_indices_ref)
topk_weights_norm_ref = (
    topk_weights_raw_ref
    / topk_weights_raw_ref.sum(dim=1, keepdim=True)
    * ROUTED_SCALING_FACTOR
)

metadata_ref = build_dispatch_metadata(
    final_topk_indices_ref.to(torch.int32),
    topk_weights_norm_ref,
    N_EXPERTS,
)
permuted_hidden_ref = permute_hidden_cpu(hidden_states, metadata_ref.permute_indices)
expert_output_ref = reference_multi_expert_deepseek(
    permuted_hidden_ref,
    metadata_ref.expert_ids_flat,
    gate_up_proj,
    down_proj,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
)
routed_output_ref = combine_routed_outputs_cpu(
    expert_output_ref,
    metadata_ref,
    N_TOKENS,
)
shared_output_ref = reference_shared_expert_deepseek(
    hidden_states,
    shared_gate_up_proj,
    shared_down_proj,
)
moe_output_ref = routed_output_ref + shared_output_ref
print(f"===== DeepSeek-V3 MoE (mode={MODE}) =====")
print(
    "Config: "
    f"N={N_TOKENS}, H={HIDDEN_SIZE}, E={N_EXPERTS}, G={N_GROUP}, "
    f"topk_group={TOPK_GROUP}, top_k={TOP_K}, I={INTERMEDIATE_SIZE}, "
    f"shared={N_SHARED_EXPERTS}, "
    f"blocks=(M{BLOCK_M}, H{BLOCK_H}, E{BLOCK_E}, I{BLOCK_I}), seed={SEED}"
)
print("===== Comparison =====")

all_ok = check_tensor("MoE output", moe_output, moe_output_ref)
print(f"Results match: {all_ok}")

if not all_ok:
    raise AssertionError("DeepSeek-V3 MoE output does not match the CPU reference")
