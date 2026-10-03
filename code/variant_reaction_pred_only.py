# -*- coding: utf-8 -*-
"""
ABLATION  -  reaction classification head trained independently
"""

from gen_metabolite_base import VariantConfig, run_pipeline

VARIANT = VariantConfig(
    name="reaction_pred_only",
    description="Reaction classification head trained on its own; no SOM head, no candidate generation, no reranker",
    gnn='gatv2',
    feature_mode='full',
    train_som=False,
    train_rxn=True,
    use_reranker=False,
    use_diversity=False,
    generate_metabolites=False,
    selection_metric='rxn_f1',
)

if __name__ == "__main__":
    run_pipeline(VARIANT)
