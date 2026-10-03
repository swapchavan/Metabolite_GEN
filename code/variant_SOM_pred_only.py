# -*- coding: utf-8 -*-
"""
ABLATION  -  site-of-metabolism head trained independently
"""

from gen_metabolite_base import VariantConfig, run_pipeline

VARIANT = VariantConfig(
    name="SOM_pred_only",
    description="SOM prediction head trained on its own; no reaction head, no candidate generation, no reranker",
    gnn='gatv2',
    feature_mode='full',
    train_som=True,
    train_rxn=False,
    use_reranker=False,
    use_diversity=False,
    generate_metabolites=False,
    selection_metric='som_mrr',
)

if __name__ == "__main__":
    run_pipeline(VARIANT)
