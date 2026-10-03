# -*- coding: utf-8 -*-
"""
ABLATION  -  simplified GNN encoder with the same rule-based generation
"""

from gen_metabolite_base import VariantConfig, run_pipeline

VARIANT = VariantConfig(
    name="full_model_simple_GNN",
    description="Simplified 2-layer GCN encoder (no attention, no edge conditioning) with the identical rule engine and reranker",
    gnn='simple_gcn',
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
