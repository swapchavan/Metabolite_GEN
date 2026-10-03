# -*- coding: utf-8 -*-
"""
ABLATION  -  engineered metabolic atom descriptors removed
"""

from gen_metabolite_base import VariantConfig, run_pipeline

VARIANT = VariantConfig(
    name="full_model_no_engineered_descr",
    description="Full model trained on the 10 generic atom descriptors only; the 9 engineered metabolic soft-spot descriptors are removed",
    gnn='gatv2',
    feature_mode='no_engineered',
    train_som=True,
    train_rxn=True,
    use_reranker=True,
    use_diversity=True,
    generate_metabolites=True,
    selection_metric='joint',
)

if __name__ == "__main__":
    run_pipeline(VARIANT)
