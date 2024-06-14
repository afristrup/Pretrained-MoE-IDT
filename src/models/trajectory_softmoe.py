import os
from copy import deepcopy

import torch
from torch import nn

from src.models.soft_moe_utils.soft_moe import Experts, FeedForward as Expert
from src.models.soft_moe_utils.distributed import all_gather_variable_dim


def construct_model(
    num_experts,
    tokens_per_expert,
    dim,
) -> torch.nn.Module:
    model = Experts(
        num_experts=num_experts,
        tokens_per_expert=tokens_per_expert,
        dim=dim,
        num_heads=3,
        num_layers=4,
        num_positions=128,
        mlp_dim=dim,
        num_classes=2,
        experts=Expert,
    )

    return model
