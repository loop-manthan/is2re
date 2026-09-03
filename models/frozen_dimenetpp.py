"""Frozen-backbone DimeNet++ variant for IS2RE fine-tuning.

Registers a thin subclass of the stock ``dimenetplusplus`` model that freezes the
message-passing backbone (``emb`` + ``interaction_blocks``) in ``__init__`` so the
freeze is in effect *before* the optimizer is constructed. Only the ``output_blocks``
(the per-block energy readout layers) remain trainable. ``rbf`` / ``sbf`` are
non-parametric (buffers only) and need no freezing.
"""

from __future__ import annotations

import fairchem.core.models.dimenet_plus_plus as _dpp  # noqa: F401  (registers "dimenetplusplus")
from fairchem.core.common.registry import registry
from fairchem.core.models.dimenet_plus_plus import DimeNetPlusPlusWrap

FROZEN_BACKBONE_MODULES = ("rbf", "emb", "interaction_blocks")


@registry.register_model("dimenetplusplus_frozen_backbone")
class DimeNetPlusPlusFrozenBackbone(DimeNetPlusPlusWrap):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        for module_name in FROZEN_BACKBONE_MODULES:
            module = getattr(self, module_name)
            for param in module.parameters():
                param.requires_grad = False
