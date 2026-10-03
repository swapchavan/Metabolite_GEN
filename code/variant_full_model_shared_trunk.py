# -*- coding: utf-8 -*-
"""
SHARED GATv2 TRUNK, TWO LIGHTWEIGHT HEADS, WEIGHTED MULTI-TASK LOSS
"""

from gen_metabolite_base import VariantConfig
from gen_metabolite_shared import run_shared_pipeline

VARIANT = VariantConfig(
    name="full_model_shared_trunk",
    description=("Shared GATv2 trunk with two lightweight task heads, trained on a "
                 "weighted sum of the SOM and reaction losses (genuine multi-task "
                 "representation learning)"),
    gnn="gatv2",
    feature_mode="full",
    train_som=True,
    train_rxn=True,
    use_reranker=True,
    use_diversity=True,
    generate_metabolites=True,
    selection_metric="joint",
)

if __name__ == "__main__":
    run_shared_pipeline(VARIANT)
