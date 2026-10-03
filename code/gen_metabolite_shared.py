# -*- coding: utf-8 -*-
"""
SHARED-TRUNK MULTI-TASK VARIANT FOR METABOLITE-GEN
"""

import os
import copy
import json
import math
from dataclasses import asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool

import joblib
from rdkit import Chem

from gen_metabolite_base import (
    # config / infra
    VariantConfig, build_argparser, set_feature_mode, set_fragment_policy,
    count_multifragment_ground_truth, enable_reproducibility,
    ensure_dir, write_log, resolve_amp, _autocast_ctx, _make_grad_scaler,
    detach_to_numpy, active_feature_names, active_feature_dim,
    EDGE_FEAT_DIM, drop_edge,
    # data
    load_dataset, scaffold_test_split, scaffold_kfold_split,
    build_parent_scaffold_df, augment_minority_classes, compute_class_weights,
    build_scaler, apply_scaler, make_loader,
    # losses
    BinaryFocalLoss, MultiClassFocalLoss,
    # downstream pipeline
    generate_candidates, rank_candidates, build_reranker, reranker_loss,
    reranked_metrics_from_predictions, som_metrics_from_predictions,
    collect_som_predictions, som_ranking_metrics, calibrate_threshold,
    # reporting
    save_som_csv, save_rxn_csv, save_metabolites_csv, save_all_images,
    fold_composition_report, resolve_final_epochs, write_cv_summary,
)

from sklearn.metrics import f1_score as sk_f1


# ═════════════════════════════════════════════════════════════════
# THE SHARED-TRUNK MODEL
# ═════════════════════════════════════════════════════════════════
class SharedTrunkMultiTask(nn.Module):

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2,
                 heads: int = 5, dropout: float = 0.1, num_classes: int = 12,
                 edge_dim: int = EDGE_FEAT_DIM, drop_edge_p: float = 0.1,
                 head_hidden: int = 64):
        super().__init__()
        self.dropout = dropout
        self.drop_edge_p = drop_edge_p

        self.convs = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            self.convs.append(GATv2Conv(last, hidden, heads=heads,
                                        dropout=dropout, add_self_loops=True,
                                        edge_dim=edge_dim))
            last = hidden * heads
        self.embed_dim = last

        # task-specific heads
        self.som_head = nn.Linear(last, 1)
        self.rxn_head = nn.Sequential(
            nn.Linear(last, head_hidden), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes))

    def encode(self, x, edge_index, edge_attr=None):
        ei, ea = drop_edge(edge_index, edge_attr, self.drop_edge_p, self.training)
        for conv in self.convs:
            x = F.dropout(F.elu(conv(x, ei, edge_attr=ea)),
                          p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, batch_vec, edge_attr=None):
        h = self.encode(x, edge_index, edge_attr)
        som_logits = self.som_head(h).squeeze(-1)
        rxn_logits = self.rxn_head(global_mean_pool(h, batch_vec))
        return som_logits, rxn_logits

    # parameter accounting, reported in the run log so the reader can verify
    # that the trunk really does hold most of the capacity
    def parameter_split(self) -> Dict[str, int]:
        trunk = sum(p.numel() for p in self.convs.parameters())
        som = sum(p.numel() for p in self.som_head.parameters())
        rxn = sum(p.numel() for p in self.rxn_head.parameters())
        return {"trunk": trunk, "som_head": som, "rxn_head": rxn,
                "total": trunk + som + rxn,
                "shared_fraction": round(trunk / max(trunk + som + rxn, 1), 4)}


class _SomView(nn.Module):

    def __init__(self, shared: SharedTrunkMultiTask):
        super().__init__()
        self.shared = shared

    def forward(self, x, edge_index, edge_attr=None):
        h = self.shared.encode(x, edge_index, edge_attr)
        return self.shared.som_head(h).squeeze(-1)

    def train(self, mode: bool = True):
        self.shared.train(mode)
        return super().train(mode)

    def eval(self):
        self.shared.eval()
        return super().eval()


# ═════════════════════════════════════════════════════════════════
# COMBINED GRAPHS: BOTH LABELS ON ONE OBJECT
# ═════════════════════════════════════════════════════════════════
def merge_graph_lists(sg_list: List[Data], rg_list: List[Data]) -> List[Data]:
    if len(sg_list) != len(rg_list):
        raise ValueError(f"graph lists differ in length: {len(sg_list)} vs {len(rg_list)}")

    out: List[Data] = []
    for i, (sg, rg) in enumerate(zip(sg_list, rg_list)):
        if sg.x.shape != rg.x.shape:
            raise ValueError(f"graph {i}: node feature shapes differ "
                             f"{tuple(sg.x.shape)} vs {tuple(rg.x.shape)}")
        if str(sg.smiles) != str(rg.smiles):
            raise ValueError(f"graph {i}: SMILES mismatch between the two lists")

        d = Data(x=sg.x.clone(),
                 edge_index=sg.edge_index.clone(),
                 edge_attr=(sg.edge_attr.clone() if getattr(sg, "edge_attr", None) is not None else None),
                 y=sg.y.clone())
        d.y_rxn = rg.y.view(-1).clone().long()
        d.smiles = sg.smiles
        d.smiles_mapped = getattr(sg, "smiles_mapped", sg.smiles)
        d.successor_smiles = getattr(sg, "successor_smiles", "")
        d.gt_metabolites_json = getattr(sg, "gt_metabolites_json", "[]")
        d.entry_id = getattr(sg, "entry_id", -1)
        out.append(d)
    return out


def scale_combined(graphs: List[Data], scaler) -> List[Data]:
    """Standardise node features, preserving y_rxn and all metadata."""
    out = []
    for g in graphs:
        g2 = Data(x=torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float),
                  edge_index=g.edge_index,
                  edge_attr=getattr(g, "edge_attr", None),
                  y=g.y)
        g2.y_rxn = g.y_rxn
        g2.smiles = g.smiles
        g2.smiles_mapped = getattr(g, "smiles_mapped", g.smiles)
        g2.successor_smiles = getattr(g, "successor_smiles", "")
        g2.gt_metabolites_json = getattr(g, "gt_metabolites_json", "[]")
        g2.entry_id = getattr(g, "entry_id", -1)
        out.append(g2)
    return out


def apply_scaler_combined(graphs: List[Data], scaler) -> None:
    """In-place feature scaling (y_rxn untouched)."""
    for g in graphs:
        g.x = torch.tensor(scaler.transform(g.x.cpu().numpy()), dtype=torch.float)


# ═════════════════════════════════════════════════════════════════
# REACTION-SIDE EVALUATION 
# ═════════════════════════════════════════════════════════════════
@torch.no_grad()
def collect_shared_rxn_predictions(model, loader, device, le, top_k=5,
                                   use_amp: bool = False) -> List[Dict]:
    """Reaction predictions in the record format the downstream pipeline expects."""
    model.eval()
    amp_enabled = resolve_amp(device, use_amp)
    results = []
    for batch in loader:
        batch = batch.to(device, non_blocking=(device.type == "cuda"))
        ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
        with _autocast_ctx(device.type, amp_enabled):
            _, logits = model(batch.x, batch.edge_index, batch.batch, ea)
            probs = torch.softmax(logits, dim=-1)
        probs = detach_to_numpy(probs)
        trues = detach_to_numpy(batch.y_rxn.view(-1)).astype(int)
        smi_l = batch.smiles if isinstance(batch.smiles, list) else [batch.smiles]
        eid_l = batch.entry_id if hasattr(batch, "entry_id") else [-1] * len(trues)
        for i in range(len(trues)):
            order = np.argsort(-probs[i])
            rec = {"smiles": smi_l[i] if i < len(smi_l) else "",
                   "entry_id": int(eid_l[i]) if i < len(eid_l) else -1,
                   "ground_truth_rxn": le.inverse_transform([int(trues[i])])[0],
                   "probs": probs[i], "order": order}
            for k in range(1, top_k + 1):
                rec[f"top{k}_rxn"] = (le.inverse_transform([int(order[k - 1])])[0]
                                      if (k - 1) < len(order) else None)
            results.append(rec)
    return results


@torch.no_grad()
def shared_rxn_evaluate(model, loader, criterion, device, le,
                        use_amp: bool = False) -> Dict:
    """Weighted and per-class F1 for the reaction head."""
    model.eval()
    amp_enabled = resolve_amp(device, use_amp)
    tl = tn = 0
    ap, at = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=(device.type == "cuda"))
        ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
        with _autocast_ctx(device.type, amp_enabled):
            _, logits = model(batch.x, batch.edge_index, batch.batch, ea)
            y = batch.y_rxn.view(-1)
            loss = criterion(logits, y)
        tl += float(loss.item()) * y.size(0); tn += y.size(0)
        ap.extend(detach_to_numpy(logits.argmax(1)).tolist())
        at.extend(detach_to_numpy(y).tolist())
    ap = np.array(ap); at = np.array(at)
    f1_per = sk_f1(at, ap, average=None, zero_division=0,
                   labels=list(range(len(le.classes_))))
    return {"loss": tl / tn if tn else float("nan"),
            "acc": float((ap == at).mean()) if len(at) else float("nan"),
            "f1": float(sk_f1(at, ap, average="weighted", zero_division=0)),
            "f1_macro": float(sk_f1(at, ap, average="macro", zero_division=0)),
            "f1_per": {le.classes_[i]: float(f1_per[i]) for i in range(len(le.classes_))}}


@torch.no_grad()
def shared_rxn_report(model, loader, device, le, outdir, prefix="test",
                      use_amp: bool = False) -> Dict:
    """Per-class precision/recall/F1/support and confusion matrix."""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    model.eval()
    amp_enabled = resolve_amp(device, use_amp)
    yp, yt = [], []
    for batch in loader:
        batch = batch.to(device, non_blocking=(device.type == "cuda"))
        ea = batch.edge_attr if hasattr(batch, "edge_attr") else None
        with _autocast_ctx(device.type, amp_enabled):
            _, logits = model(batch.x, batch.edge_index, batch.batch, ea)
        yp.extend(detach_to_numpy(logits.argmax(1)).tolist())
        yt.extend(detach_to_numpy(batch.y_rxn.view(-1)).tolist())
    yp = np.array(yp); yt = np.array(yt)
    labels = list(range(len(le.classes_)))
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    pd.DataFrame({"reaction_class": list(le.classes_),
                  "precision": np.round(p, 4), "recall": np.round(r, 4),
                  "f1": np.round(f, 4), "support": s}).to_csv(
        os.path.join(outdir, f"{prefix}_rxn_per_class_report.csv"), index=False)
    pd.DataFrame(confusion_matrix(yt, yp, labels=labels),
                 index=list(le.classes_), columns=list(le.classes_)).to_csv(
        os.path.join(outdir, f"{prefix}_rxn_confusion_matrix.csv"))
    summary = {"accuracy": float((yp == yt).mean()) if len(yt) else float("nan"),
               "f1_weighted": float(sk_f1(yt, yp, average="weighted", zero_division=0)),
               "f1_macro": float(sk_f1(yt, yp, average="macro", zero_division=0)),
               "n": int(len(yt))}
    with open(os.path.join(outdir, f"{prefix}_rxn_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


# ═════════════════════════════════════════════════════════════════
# TRAINING STEPS
# ═════════════════════════════════════════════════════════════════
def train_shared_epoch(model, loader, optimizer, focal_som, focal_rxn,
                       scaler, device, amp_enabled,
                       w_som: float, w_rxn: float) -> Dict[str, float]:
    """
    One epoch of joint multi-task training.

    A single forward pass yields both outputs; the weighted sum is backpropagated
    once, so both task gradients accumulate in the shared trunk before the
    optimiser step. This is the operative difference from the reference model,
    where the two losses never meet.
    """
    model.train()
    tot = tot_s = tot_r = 0.0
    nb = 0
    for b in loader:
        b = b.to(device, non_blocking=(device.type == "cuda"))
        ea = b.edge_attr if hasattr(b, "edge_attr") else None
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device.type, amp_enabled):
            som_logits, rxn_logits = model(b.x, b.edge_index, b.batch, ea)
            l_som = focal_som(som_logits, b.y.view(-1))
            l_rxn = focal_rxn(rxn_logits, b.y_rxn.view(-1))
            loss = w_som * l_som + w_rxn * l_rxn
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        tot += float(loss.item()); tot_s += float(l_som.item()); tot_r += float(l_rxn.item())
        nb += 1
    n = max(nb, 1)
    return {"loss": tot / n, "loss_som": tot_s / n, "loss_rxn": tot_r / n}


def train_reranker_epoch_shared(model, rr, eval_loader, optimizer, scaler,
                                device, amp_enabled, args, diversity_weight, le) -> float:
    """Reranker epoch, driven by predictions from the shared model."""
    model.eval(); rr.train()
    som_res = collect_som_predictions(_SomView(model), eval_loader, device, use_amp=amp_enabled)
    rxn_res = collect_shared_rxn_predictions(model, eval_loader, device, le, use_amp=amp_enabled)

    total = torch.tensor(0.0, device=device); nr = 0
    for sr, rres in zip(som_res, rxn_res):
        cands = generate_candidates(sr["smiles"], sr, rres, le, args.top_som, args.top_rxn)
        if not cands:
            continue
        n = len(cands)
        tan_v = torch.tensor([c["tanimoto_gt"] for c in cands], dtype=torch.float, device=device)
        som_pv = torch.tensor([c["som_prob"] for c in cands], dtype=torch.float, device=device)
        rxn_pv = torch.tensor([c["rxn_prob"] for c in cands], dtype=torch.float, device=device)
        fps_np = np.stack([c["ecfp4_vec"] for c in cands], 0)
        fp_m = torch.tensor(fps_np, dtype=torch.float, device=device)
        dot = fps_np @ fps_np.T
        ssum = fps_np.sum(axis=1, keepdims=True)
        tan_mat = dot / np.clip(ssum + ssum.T - dot, 1e-8, None)
        np.fill_diagonal(tan_mat, 0.0)
        inter_v = torch.tensor(tan_mat.sum(axis=1) / max(n - 1, 1),
                               dtype=torch.float, device=device)
        max_rank = max(max(c["som_rank"] for c in cands), 1)
        srn_v = torch.tensor([c["som_rank"] / max_rank for c in cands], dtype=torch.float, device=device)
        rrn_v = torch.tensor([c["rxn_rank"] / max_rank for c in cands], dtype=torch.float, device=device)
        total = total + reranker_loss(rr, som_pv, rxn_pv, tan_v, fp_m, inter_v,
                                      srn_v, rrn_v, device,
                                      diversity_weight=diversity_weight)
        nr += 1

    if nr > 0:
        optimizer.zero_grad(set_to_none=True)
        loss = total / nr
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(rr.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(rr.parameters(), 5.0)
            optimizer.step()
        return float(total.item() / nr)
    return float("nan")


def build_shared_model(args, in_dim: int, num_classes: int, device,
                       drop_edge_p: Optional[float] = None) -> SharedTrunkMultiTask:
    return SharedTrunkMultiTask(
        in_dim,
        hidden=args.trunk_hidden,
        layers=args.trunk_layers,
        heads=args.heads,
        dropout=args.dropout,
        num_classes=num_classes,
        drop_edge_p=(args.drop_edge_p if drop_edge_p is None else drop_edge_p),
        head_hidden=args.head_hidden,
    ).to(device)


# ═════════════════════════════════════════════════════════════════
# CROSS-VALIDATION FOLD
# ═════════════════════════════════════════════════════════════════
def train_one_fold_shared(variant, fold_id, train_sg, val_sg, train_rg, val_rg,
                          raw_train_df, args, device, cv_dir, le, num_classes):
    fold_seed = int(args.seed) + int(fold_id)
    enable_reproducibility(fold_seed)

    fold_dir = os.path.join(cv_dir, f"fold_{fold_id}")
    ensure_dir(fold_dir)
    log_fp = os.path.join(fold_dir, "training_log.txt")

    train_sg = [copy.deepcopy(g) for g in train_sg]
    train_rg = [copy.deepcopy(g) for g in train_rg]
    val_sg = [copy.deepcopy(g) for g in val_sg]
    val_rg = [copy.deepcopy(g) for g in val_rg]

    # augment on the parallel lists first, then merge, so augmentation is
    # byte-identical to the reference variant
    if args.aug_factor > 0:
        train_sg, train_rg = augment_minority_classes(
            train_sg, train_rg, raw_train_df, le,
            aug_threshold=args.aug_threshold, aug_factor=args.aug_factor)

    train_c = merge_graph_lists(train_sg, train_rg)
    val_c = merge_graph_lists(val_sg, val_rg)

    sc = build_scaler(train_c)
    apply_scaler_combined(train_c, sc)
    apply_scaler_combined(val_c, sc)

    bs, sd = args.batch_size, fold_seed
    pin = (device.type == "cuda")
    tl = make_loader(train_c, bs, True, sd, args.num_workers, pin)
    tl_e = make_loader(train_c, bs, False, sd, args.num_workers, pin)
    vl = make_loader(val_c, bs, False, sd, args.num_workers, pin)

    in_dim = train_c[0].x.size(-1)
    model = build_shared_model(args, in_dim, num_classes, device)
    rr = build_reranker(args, device)

    psplit = model.parameter_split()
    msg = (f"[{variant.name}|Fold {fold_id}] shared trunk: {psplit['trunk']:,} params "
           f"({psplit['shared_fraction']:.1%} of {psplit['total']:,}); "
           f"SOM head {psplit['som_head']:,}, reaction head {psplit['rxn_head']:,}")
    print(msg); write_log(log_fp, msg)

    amp_enabled = resolve_amp(device, bool(args.amp))
    sc_main = _make_grad_scaler(amp_enabled)
    sc_rer = _make_grad_scaler(amp_enabled)

    focal_som = BinaryFocalLoss(args.alpha, args.gamma)
    focal_rxn = MultiClassFocalLoss(weight=compute_class_weights(raw_train_df, le, device),
                                    gamma=args.focal_gamma_rxn)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_rr = torch.optim.AdamW(rr.parameters(), lr=args.lr_rer, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr_min)

    w_som, w_rxn = float(args.loss_weight_som), float(args.loss_weight_rxn)
    div_w = args.diversity_weight if variant.use_diversity else 0.0

    best_state = best_rr_state = None
    best_score = -1.0; best_ep = -1; nim = 0
    best_som_mrr = best_rxn_f1 = best_joint = -1.0
    history: List[Dict] = []

    for ep in range(1, args.epochs + 1):
        tr = train_shared_epoch(model, tl, opt, focal_som, focal_rxn,
                                sc_main, device, amp_enabled, w_som, w_rxn)
        sch.step()
        rer_loss = train_reranker_epoch_shared(model, rr, tl_e, opt_rr, sc_rer,
                                               device, amp_enabled, args, div_w, le)

        vm = som_ranking_metrics(_SomView(model), vl, device, focal_som, use_amp=amp_enabled)
        vr = shared_rxn_evaluate(model, vl, focal_rxn, device, le, use_amp=amp_enabled)
        best_som_mrr = max(best_som_mrr, float(vm["mrr"]) if not math.isnan(vm["mrr"]) else -1.0)
        best_rxn_f1 = max(best_rxn_f1, float(vr["f1"]))

        s_res = collect_som_predictions(_SomView(model), vl, device, use_amp=amp_enabled)
        r_res = collect_shared_rxn_predictions(model, vl, device, le, use_amp=amp_enabled)
        jm = reranked_metrics_from_predictions(
            s_res, r_res, rr, device, le,
            top_som=args.top_som, top_rxn=args.top_rxn,
            tanimoto_threshold=args.tanimoto_threshold,
            use_reranker=variant.use_reranker, use_amp=amp_enabled)
        best_joint = max(best_joint, float(jm["joint"]))
        score = float(jm["joint"])

        ln = (f"[{variant.name}|Fold {fold_id}] Ep {ep:03d} lr={sch.get_last_lr()[0]:.2e} | "
              f"L={tr['loss']:.4f} (som {tr['loss_som']:.4f} x{w_som} / "
              f"rxn {tr['loss_rxn']:.4f} x{w_rxn}) | "
              f"SOM top1={vm['top1']:.3f} mrr={vm['mrr']:.3f} | "
              f"Rxn acc={vr['acc']:.3f} f1={vr['f1']:.3f} macro={vr['f1_macro']:.3f} | "
              f"Rer={rer_loss:.4f} | Met top1={jm['top1']:.3f} joint={jm['joint']:.4f} "
              f"(n={jm['n_molecules']} empty={jm['n_empty']})")
        print(ln); write_log(log_fp, ln)

        history.append({"epoch": ep, "loss": tr["loss"], "loss_som": tr["loss_som"],
                        "loss_rxn": tr["loss_rxn"], "som_top1": vm["top1"],
                        "som_mrr": vm["mrr"], "rxn_acc": vr["acc"], "rxn_f1": vr["f1"],
                        "rxn_f1_macro": vr["f1_macro"], "met_top1": jm["top1"],
                        "met_mrr": jm["mrr"], "met_joint": jm["joint"],
                        "reranker_loss": None if math.isnan(rer_loss) else rer_loss,
                        "selection_score": score})

        if score > best_score:
            best_score = score; best_ep = ep; nim = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_rr_state = {k: v.cpu().clone() for k, v in rr.state_dict().items()}
        else:
            nim += 1
        if nim >= args.patience:
            m = (f"[{variant.name}|Fold {fold_id}] Early stop @ epoch {ep} "
                 f"(joint stale {nim} epochs; best={best_score:.4f} @ ep {best_ep})")
            print(m); write_log(log_fp, m); break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    if best_rr_state:
        rr.load_state_dict({k: v.to(device) for k, v in best_rr_state.items()})

    torch.save({"model": model.state_dict(), "in_dim": in_dim,
                "trunk_hidden": args.trunk_hidden, "trunk_layers": args.trunk_layers,
                "heads": args.heads, "head_hidden": args.head_hidden,
                "dropout": args.dropout, "drop_edge_p": args.drop_edge_p,
                "num_classes": num_classes, "loss_weight_som": w_som,
                "loss_weight_rxn": w_rxn, "parameter_split": psplit,
                "feature_names": active_feature_names()},
               os.path.join(fold_dir, "shared_model.pt"))
    torch.save({"model": rr.state_dict(), "fp_dim": 2048, "hidden": args.hidden_rer,
                "dropout": args.dropout, "diversity_weight": div_w},
               os.path.join(fold_dir, "reranker.pt"))
    joblib.dump(sc, os.path.join(fold_dir, "feature_scaler.pkl"))
    pd.DataFrame(history).to_csv(os.path.join(fold_dir, "epoch_history.csv"), index=False)

    # validation artefacts
    yt, yp = [], []
    with torch.no_grad():
        sv = _SomView(model); sv.eval()
        for b in vl:
            b = b.to(device, non_blocking=(device.type == "cuda"))
            ea = b.edge_attr if hasattr(b, "edge_attr") else None
            with _autocast_ctx(device.type, amp_enabled):
                prob = torch.sigmoid(sv(b.x, b.edge_index, ea))
            yt.append(detach_to_numpy(b.y.view(-1))); yp.append(detach_to_numpy(prob))
    thr = calibrate_threshold(np.concatenate(yt), np.concatenate(yp))
    open(os.path.join(fold_dir, "best_threshold.txt"), "w").write(str(thr))

    vs_res = collect_som_predictions(_SomView(model), vl, device, use_amp=amp_enabled)
    vr_res = collect_shared_rxn_predictions(model, vl, device, le, use_amp=amp_enabled)
    save_som_csv(vs_res, os.path.join(fold_dir, "val_som_predictions.csv"))
    save_rxn_csv(vr_res, os.path.join(fold_dir, "val_rxn_predictions.csv"))
    shared_rxn_report(model, vl, device, le, fold_dir, prefix="val", use_amp=amp_enabled)

    final_val = reranked_metrics_from_predictions(
        vs_res, vr_res, rr, device, le,
        top_som=args.top_som, top_rxn=args.top_rxn,
        tanimoto_threshold=args.tanimoto_threshold,
        use_reranker=variant.use_reranker, use_amp=amp_enabled, collect_rows=True)
    pd.DataFrame(final_val.pop("_rows", [])).to_csv(
        os.path.join(fold_dir, "val_error_analysis.csv"), index=False)

    v_mc, v_rl = [], []
    for sr, rres in zip(vs_res, vr_res):
        cands = generate_candidates(sr["smiles"], sr, rres, le, args.top_som, args.top_rxn)
        ranked = rank_candidates(rr, cands, device, variant.use_reranker, use_amp=amp_enabled)
        v_mc.append((sr["smiles"], sr.get("successor_smiles", ""), ranked)); v_rl.append(ranked)
    save_metabolites_csv(v_mc, os.path.join(fold_dir, "val_metabolites.csv"), top_n=5)
    if args.save_images:
        img_dir = os.path.join(fold_dir, "images"); ensure_dir(img_dir)
        save_all_images(vs_res, v_rl, img_dir, args.subimg_w, args.subimg_h, top_n=5)

    final_som = som_ranking_metrics(_SomView(model), vl, device, focal_som, use_amp=amp_enabled)
    final_rxn = shared_rxn_evaluate(model, vl, focal_rxn, device, le, use_amp=amp_enabled)

    metrics = {
        "fold": int(fold_id), "selection_metric": variant.selection_metric,
        "best_selection_score": float(best_score), "best_epoch": int(best_ep),
        "val_som": {k: float(v) for k, v in final_som.items() if isinstance(v, (int, float))},
        "val_rxn": {k: v for k, v in final_rxn.items() if k != "f1_per"},
        "val_rxn_f1_per_class": final_rxn.get("f1_per", {}),
        "val_metabolite": final_val,
        "best_som_mrr_seen": float(best_som_mrr),
        "best_rxn_f1_seen": float(best_rxn_f1),
        "best_joint_seen": float(best_joint),
        "loss_weights": {"som": w_som, "rxn": w_rxn},
        "parameter_split": psplit,
        "n_train_graphs": int(len(train_c)), "n_val_graphs": int(len(val_c)),
    }
    with open(os.path.join(fold_dir, "fold_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return {"metrics": metrics, "best_epoch": int(best_ep),
            "best_selection_score": float(best_score),
            "scaler": sc, "model_state": best_state, "rer_state": best_rr_state,
            "in_dim": in_dim}


# ═════════════════════════════════════════════════════════════════
# FINAL MODEL ON THE COMPLETE DEVELOPMENT SET
# ═════════════════════════════════════════════════════════════════
def train_full_model_shared(variant, dev_sg, dev_rg, raw_dev_df, n_epochs,
                            args, device, out_dir, le, num_classes):
    enable_reproducibility(int(args.seed))
    ensure_dir(out_dir)
    log_fp = os.path.join(out_dir, "training_log.txt")

    dev_sg = [copy.deepcopy(g) for g in dev_sg]
    dev_rg = [copy.deepcopy(g) for g in dev_rg]
    if args.aug_factor > 0:
        dev_sg, dev_rg = augment_minority_classes(
            dev_sg, dev_rg, raw_dev_df, le,
            aug_threshold=args.aug_threshold, aug_factor=args.aug_factor)
    dev_c = merge_graph_lists(dev_sg, dev_rg)

    sc = build_scaler(dev_c); apply_scaler_combined(dev_c, sc)
    pin = (device.type == "cuda")
    tl = make_loader(dev_c, args.batch_size, True, args.seed, args.num_workers, pin)
    tl_e = make_loader(dev_c, args.batch_size, False, args.seed, args.num_workers, pin)

    in_dim = dev_c[0].x.size(-1)
    model = build_shared_model(args, in_dim, num_classes, device)
    rr = build_reranker(args, device)
    psplit = model.parameter_split()

    hdr = (f"[{variant.name}|FULL] complete development set: {len(dev_c)} graphs, "
           f"{n_epochs} epochs; shared trunk holds {psplit['shared_fraction']:.1%} "
           f"of {psplit['total']:,} parameters")
    print(hdr); write_log(log_fp, hdr)

    amp_enabled = resolve_amp(device, bool(args.amp))
    sc_main = _make_grad_scaler(amp_enabled)
    sc_rer = _make_grad_scaler(amp_enabled)
    focal_som = BinaryFocalLoss(args.alpha, args.gamma)
    focal_rxn = MultiClassFocalLoss(weight=compute_class_weights(raw_dev_df, le, device),
                                    gamma=args.focal_gamma_rxn)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    opt_rr = torch.optim.AdamW(rr.parameters(), lr=args.lr_rer, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=args.lr_min)

    w_som, w_rxn = float(args.loss_weight_som), float(args.loss_weight_rxn)
    div_w = args.diversity_weight if variant.use_diversity else 0.0
    history = []

    for ep in range(1, n_epochs + 1):
        tr = train_shared_epoch(model, tl, opt, focal_som, focal_rxn,
                                sc_main, device, amp_enabled, w_som, w_rxn)
        sch.step()
        rer_loss = train_reranker_epoch_shared(model, rr, tl_e, opt_rr, sc_rer,
                                               device, amp_enabled, args, div_w, le)
        tm = som_ranking_metrics(_SomView(model), tl_e, device, focal_som, use_amp=amp_enabled)
        trx = shared_rxn_evaluate(model, tl_e, focal_rxn, device, le, use_amp=amp_enabled)
        ln = (f"[{variant.name}|FULL] Ep {ep:03d}/{n_epochs} L={tr['loss']:.4f} "
              f"(som {tr['loss_som']:.4f} / rxn {tr['loss_rxn']:.4f}) | "
              f"SOM top1={tm['top1']:.3f} | Rxn f1={trx['f1']:.3f} | Rer={rer_loss:.4f}"
              "   [train-set diagnostics]")
        print(ln); write_log(log_fp, ln)
        history.append({"epoch": ep, **tr, "train_som_top1": tm["top1"],
                        "train_rxn_f1": trx["f1"],
                        "reranker_loss": None if math.isnan(rer_loss) else rer_loss})

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "epoch_history.csv"), index=False)
    torch.save({"model": model.state_dict(), "in_dim": in_dim,
                "trunk_hidden": args.trunk_hidden, "trunk_layers": args.trunk_layers,
                "heads": args.heads, "head_hidden": args.head_hidden,
                "dropout": args.dropout, "num_classes": num_classes,
                "loss_weight_som": w_som, "loss_weight_rxn": w_rxn,
                "parameter_split": psplit, "n_epochs": n_epochs,
                "feature_names": active_feature_names()},
               os.path.join(out_dir, "shared_model_full.pt"))
    torch.save({"model": rr.state_dict(), "fp_dim": 2048, "hidden": args.hidden_rer,
                "dropout": args.dropout, "n_epochs": n_epochs},
               os.path.join(out_dir, "reranker_full.pt"))
    joblib.dump(sc, os.path.join(out_dir, "feature_scaler.pkl"))

    return {"model": model, "reranker": rr, "scaler": sc, "in_dim": in_dim,
            "criterion_som": focal_som, "criterion_rxn": focal_rxn,
            "n_epochs": n_epochs, "parameter_split": psplit}


def evaluate_split_shared(variant, bundle, sg, rg, args, device, le,
                          out_dir, split_name, save_images=False) -> Dict:
    ensure_dir(out_dir)
    amp_enabled = resolve_amp(device, bool(args.amp))
    model, rr = bundle["model"], bundle["reranker"]

    comb = scale_combined(merge_graph_lists(sg, rg), bundle["scaler"])
    loader = make_loader(comb, args.batch_size, False, args.seed, args.num_workers,
                         (device.type == "cuda"))

    som_res = collect_som_predictions(_SomView(model), loader, device, use_amp=amp_enabled)
    rxn_res = collect_shared_rxn_predictions(model, loader, device, le, use_amp=amp_enabled)
    save_som_csv(som_res, os.path.join(out_dir, f"{split_name}_som_predictions.csv"))
    save_rxn_csv(rxn_res, os.path.join(out_dir, f"{split_name}_rxn_predictions.csv"))

    metrics: Dict[str, Any] = {"split": split_name, "n_molecules": len(sg),
                               "som_only": som_metrics_from_predictions(som_res),
                               "reaction": shared_rxn_report(model, loader, device, le,
                                                             out_dir, prefix=split_name,
                                                             use_amp=amp_enabled)}

    met = reranked_metrics_from_predictions(
        som_res, rxn_res, rr, device, le,
        top_som=args.top_som, top_rxn=args.top_rxn,
        tanimoto_threshold=args.tanimoto_threshold,
        use_reranker=variant.use_reranker, use_amp=amp_enabled, collect_rows=True)
    pd.DataFrame(met.pop("_rows", [])).to_csv(
        os.path.join(out_dir, f"{split_name}_error_analysis.csv"), index=False)
    metrics["metabolite"] = met

    mc, rl_all = [], []
    for sr, rres in zip(som_res, rxn_res):
        cands = generate_candidates(sr["smiles"], sr, rres, le, args.top_som, args.top_rxn)
        ranked = rank_candidates(rr, cands, device, variant.use_reranker, use_amp=amp_enabled)
        mc.append((sr["smiles"], sr.get("successor_smiles", ""), ranked)); rl_all.append(ranked)
    save_metabolites_csv(mc, os.path.join(out_dir, f"{split_name}_metabolites.csv"), top_n=5)
    if save_images:
        img_dir = os.path.join(out_dir, f"{split_name}_images"); ensure_dir(img_dir)
        save_all_images(som_res, rl_all, img_dir, args.subimg_w, args.subimg_h, top_n=5)

    with open(os.path.join(out_dir, f"{split_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def evaluate_cv_models_on_test_shared(variant, fold_res, tst_sg, tst_rg,
                                      args, device, le, nc, cv_dir) -> Dict:
    """Each fold model, plus a probability-averaging ensemble, on the test set."""
    amp_enabled = resolve_amp(device, bool(args.amp))
    pin = (device.type == "cuda")
    base_comb = merge_graph_lists(tst_sg, tst_rg)
    per_som, per_rxn, rows, rerankers = [], [], [], []

    for i, fold in enumerate(fold_res, 1):
        model = build_shared_model(args, fold["in_dim"], nc, device, drop_edge_p=0.0)
        model.load_state_dict({k: v.to(device) for k, v in fold["model_state"].items()})
        loader = make_loader(scale_combined(base_comb, fold["scaler"]),
                             args.batch_size, False, args.seed, args.num_workers, pin)
        s_res = collect_som_predictions(_SomView(model), loader, device, use_amp=amp_enabled)
        r_res = collect_shared_rxn_predictions(model, loader, device, le, use_amp=amp_enabled)
        per_som.append(s_res); per_rxn.append(r_res)

        rr = None
        if fold["rer_state"] is not None:
            rr = build_reranker(args, device)
            rr.load_state_dict({k: v.to(device) for k, v in fold["rer_state"].items()})
            rr.eval(); rerankers.append(rr)

        rows.append({"fold": i,
                     "som_only": som_metrics_from_predictions(s_res),
                     "metabolite": reranked_metrics_from_predictions(
                         s_res, r_res, rr, device, le,
                         top_som=args.top_som, top_rxn=args.top_rxn,
                         tanimoto_threshold=args.tanimoto_threshold,
                         use_reranker=variant.use_reranker, use_amp=amp_enabled)})

    # ensemble by averaging probabilities across fold models
    ens_som = [dict(r) for r in per_som[0]]
    for i, rec in enumerate(ens_som):
        probs = np.stack([pf[i]["probs"] for pf in per_som], 0).mean(0)
        order = np.argsort(-probs)
        rec["probs"] = probs; rec["order"] = order
        for k in range(1, 6):
            rec[f"top{k}_SOM"] = int(order[k - 1]) if (k - 1) < len(order) else None
    ens_rxn = [dict(r) for r in per_rxn[0]]
    for i, rec in enumerate(ens_rxn):
        probs = np.stack([pf[i]["probs"] for pf in per_rxn], 0).mean(0)
        order = np.argsort(-probs)
        rec["probs"] = probs; rec["order"] = order
        for k in range(1, 6):
            rec[f"top{k}_rxn"] = (le.inverse_transform([int(order[k - 1])])[0]
                                  if (k - 1) < len(order) else None)

    class _MeanReranker(nn.Module):
        def __init__(self, models): super().__init__(); self.models = nn.ModuleList(models)
        def forward(self, feat): return torch.stack([m(feat) for m in self.models], 0).mean(0)

    mean_rr = _MeanReranker(rerankers).to(device) if rerankers else None
    ens = {"som_only": som_metrics_from_predictions(ens_som),
           "metabolite": reranked_metrics_from_predictions(
               ens_som, ens_rxn, mean_rr, device, le,
               top_som=args.top_som, top_rxn=args.top_rxn,
               tanimoto_threshold=args.tanimoto_threshold,
               use_reranker=(mean_rr is not None), use_amp=amp_enabled)}
    save_som_csv(ens_som, os.path.join(cv_dir, "ensemble_test_som_predictions.csv"))
    save_rxn_csv(ens_rxn, os.path.join(cv_dir, "ensemble_test_rxn_predictions.csv"))

    def _agg(path):
        vals = []
        for r in rows:
            cur = r
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
                if cur is None: break
            if isinstance(cur, (int, float)): vals.append(float(cur))
        if not vals: return None
        return {"mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)), "max": float(np.max(vals)), "n_folds": len(vals)}

    agg = {}
    for k in ("top1", "top3", "top5", "mrr"):
        a = _agg(["som_only", k]); agg[f"som_{k}"] = a if a else None
    for k in ("top1", "top3", "top5", "mrr", "top1_exact_match", "candidate_pool_recall"):
        a = _agg(["metabolite", k]); agg[f"metabolite_{k}"] = a if a else None

    out = {"per_fold": rows, "ensemble": ens, "fold_mean_sd_on_test": agg}
    with open(os.path.join(cv_dir, "fold_models_on_test.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


# ═════════════════════════════════════════════════════════════════
# ARGUMENTS AND PIPELINE
# ═════════════════════════════════════════════════════════════════
def build_shared_argparser(variant):
    p = build_argparser(variant)
    g = p.add_argument_group("shared-trunk multi-task settings")
    g.add_argument("--trunk-hidden", type=int, default=64,
                   help="Hidden width per attention head of the SHARED trunk")
    g.add_argument("--trunk-layers", type=int, default=2,
                   help="Number of GATv2 layers in the SHARED trunk")
    g.add_argument("--head-hidden", type=int, default=64,
                   help="Hidden width of the reaction head MLP (kept small so the "
                        "trunk holds most of the capacity)")
    g.add_argument("--loss-weight-som", type=float, default=1.0,
                   help="Weight of the SOM loss in the combined objective")
    g.add_argument("--loss-weight-rxn", type=float, default=1.0,
                   help="Weight of the reaction loss in the combined objective; "
                        "the key new hyperparameter, worth sweeping")
    return p


def run_shared_pipeline(variant, argv=None):
    variant.check()
    args = build_shared_argparser(variant).parse_args(argv)
    if int(args.disable_reranker):
        variant = VariantConfig(**{**asdict(variant), "use_reranker": False})
        variant.check()
    if int(args.skip_cv) and int(args.final_epochs) <= 0:
        raise SystemExit("--skip-cv requires --final-epochs > 0")

    set_feature_mode(variant.feature_mode)
    set_fragment_policy(args.fragment_policy, args.min_fragment_atoms)
    enable_reproducibility(args.seed)
    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device.startswith("cuda") else "cpu")

    outdir = args.outdir
    cv_dir = os.path.join(outdir, "five_fold_CV")
    full_dir = os.path.join(outdir, "full_model_results")
    for d in (outdir, cv_dir, full_dir):
        ensure_dir(d)

    print("=" * 78)
    print(f"  VARIANT : {variant.name}")
    print(f"  {variant.description}")
    print(f"  SHARED GATv2 trunk: {args.trunk_layers} layers x {args.trunk_hidden} "
          f"x {args.heads} heads   heads: SOM linear, reaction MLP({args.head_hidden})")
    print(f"  combined loss: {args.loss_weight_som} * L_som + {args.loss_weight_rxn} * L_rxn")
    print(f"  features={variant.feature_mode} ({active_feature_dim()}-d)  "
          f"reranker={variant.use_reranker}  diversity={variant.use_diversity}")
    print(f"  product fragments: policy={args.fragment_policy} "
          f"(min {args.min_fragment_atoms} heavy atoms)")
    print(f"  device={device}  outdir={outdir}")
    print("=" * 78)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump({"variant": asdict(variant), "args": vars(args),
                   "atom_features_used": active_feature_names(),
                   "architecture": "shared_gatv2_trunk_multitask",
                   "fragment_policy": {"policy": args.fragment_policy,
                                       "min_heavy_atoms": args.min_fragment_atoms}},
                  f, indent=2)

    print("Loading dataset ...")
    sg, rg, le, df = load_dataset(args.csv)
    nc = len(le.classes_)
    print(f"  Molecules {len(sg)}  classes {nc}  atom features {active_feature_dim()}")
    with open(os.path.join(outdir, "label_mapping.json"), "w") as f:
        json.dump({int(i): c for i, c in enumerate(le.classes_)}, f, indent=2)
    with open(os.path.join(outdir, "ground_truth_fragment_check.json"), "w") as f:
        json.dump(count_multifragment_ground_truth(df), f, indent=2)

    pre_idx, tst_idx, holdout_info, _ = scaffold_test_split(df, 0.20, args.seed)
    pre_sg = [sg[i] for i in pre_idx]; pre_rg = [rg[i] for i in pre_idx]
    tst_sg = [sg[i] for i in tst_idx]; tst_rg = [rg[i] for i in tst_idx]
    pre_df = df.iloc[pre_idx].reset_index(drop=True)
    print(f"  Development {len(pre_sg)}  |  held-out test {len(tst_sg)}")

    parent_df = build_parent_scaffold_df(pre_df).reset_index(drop=True)
    parent_df["parent_index"] = np.arange(len(parent_df), dtype=int)
    parent_to_rows = (pre_df.assign(parent_id=pre_df["Predecessor_SMILES"].astype(str).str.strip())
                      .groupby("parent_id").indices)

    fold_res, fold_row_idx = [], []
    if not int(args.skip_cv):
        for fid, (tri_p, vli_p) in enumerate(
                scaffold_kfold_split(parent_df, args.kfolds, args.seed), 1):
            tr_ids = parent_df.iloc[tri_p]["parent_id"].tolist()
            vl_ids = parent_df.iloc[vli_p]["parent_id"].tolist()
            tri = np.array(sorted(r for pid in tr_ids for r in parent_to_rows[pid]), dtype=int)
            vli = np.array(sorted(r for pid in vl_ids for r in parent_to_rows[pid]), dtype=int)
            fold_row_idx.append((tri, vli))
            print(f"\n{'=' * 72}\n  [{variant.name}] Fold {fid}/{args.kfolds}  "
                  f"train={len(tri)} val={len(vli)}\n{'=' * 72}")
            fold_res.append(train_one_fold_shared(
                variant, fid,
                [pre_sg[i] for i in tri], [pre_sg[i] for i in vli],
                [pre_rg[i] for i in tri], [pre_rg[i] for i in vli],
                pre_df.iloc[tri].reset_index(drop=True),
                args, device, cv_dir, le, nc))
        fold_composition_report(pre_df, fold_row_idx, le,
                                os.path.join(cv_dir, "fold_composition.csv"))
        write_cv_summary(variant, fold_res, cv_dir, args)
        print("\nEvaluating each CV fold model and their ensemble on the test set ...")
        evaluate_cv_models_on_test_shared(variant, fold_res, tst_sg, tst_rg,
                                          args, device, le, nc, cv_dir)

    if int(args.skip_full):
        print("\nStopping after cross-validation (--skip-full 1).")
        return

    n_final = resolve_final_epochs(fold_res, args)
    print(f"\n{'=' * 78}\n  FINAL SHARED-TRUNK MODEL — complete development set "
          f"({len(pre_sg)} molecules), {n_final} epochs\n{'=' * 78}")
    bundle = train_full_model_shared(variant, pre_sg, pre_rg, pre_df,
                                     n_final, args, device, full_dir, le, nc)

    print("\nPredicting on the development (training) set ...")
    train_metrics = evaluate_split_shared(variant, bundle, pre_sg, pre_rg, args,
                                          device, le, full_dir, "train")
    print("Predicting on the held-out test set ...")
    test_metrics = evaluate_split_shared(variant, bundle, tst_sg, tst_rg, args,
                                         device, le, full_dir, "test",
                                         save_images=bool(args.save_images))

    with open(os.path.join(full_dir, "full_model_summary.json"), "w") as f:
        json.dump({"variant": asdict(variant),
                   "architecture": "shared_gatv2_trunk_multitask",
                   "parameter_split": bundle["parameter_split"],
                   "loss_weights": {"som": args.loss_weight_som,
                                    "rxn": args.loss_weight_rxn},
                   "final_epochs": n_final,
                   "final_epochs_source": ("--final-epochs" if int(args.final_epochs) > 0
                                           else f"CV {args.final_epochs_rule} of per-fold best epoch"),
                   "n_development_molecules": len(pre_sg),
                   "n_test_molecules": len(tst_sg),
                   "train": train_metrics, "test": test_metrics}, f, indent=2)

    print(f"\n{'=' * 78}\n  {variant.name} — HELD-OUT TEST RESULTS\n{'=' * 78}")
    for block in ("som_only", "reaction", "metabolite"):
        if block in test_metrics:
            print(f"  [{block}]")
            for k, v in test_metrics[block].items():
                print(f"    {k:26s}: {v:.4f}" if isinstance(v, float) else f"    {k:26s}: {v}")
    print(f"\nDone.\n  CV outputs        : {cv_dir}\n  Full-model outputs: {full_dir}")
