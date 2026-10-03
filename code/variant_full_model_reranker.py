# -*- coding: utf-8 -*-
"""
ABLATION  -  learned reranker WITHOUT the diversity objective
"""

from gen_metabolite_base import VariantConfig, run_pipeline

VARIANT = VariantConfig(
    name="full_model_reranker",
    description="Full model with a plain listwise reranker; the diversity-aware ranking term is removed",
    gnn='gatv2',
    feature_mode='full',
    train_som=True,
    train_rxn=True,
    use_reranker=True,
    use_diversity=False,
    generate_metabolites=True,
    selection_metric='joint',
)

if __name__ == "__main__":
    run_pipeline(VARIANT)
