# -*- coding: utf-8 -*-
"""
REFERENCE FULL MODEL  -  diversity-aware learned reranker
"""

from gen_metabolite_base import VariantConfig, run_pipeline

VARIANT = VariantConfig(
    name="full_model_diversity_reranker",
    description="Reference full model: GATv2 encoder, 19-d atom features, learned reranker with the diversity-aware ranking objective",
    gnn='gatv2',
    feature_mode='full',
    train_som=True,
    train_rxn=True,
    use_reranker=True,
    use_diversity=True,
    generate_metabolites=True,
    selection_metric='joint',
)

if __name__ == "__main__":
    run_pipeline(VARIANT)
