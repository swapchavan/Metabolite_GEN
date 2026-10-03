# -*- coding: utf-8 -*-
"""
BAYESIAN HYPERPARAMETER OPTIMISATION (GPyOpt) FOR THE REFERENCE FULL MODEL
"""

import os
import sys
import csv
import json
import copy
import time
import shutil
import argparse
import traceback
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch

from gen_metabolite_base import (
    VariantConfig,
    build_argparser,
    run_pipeline,
    set_feature_mode,
    enable_reproducibility,
    ensure_dir,
    write_log,
    load_dataset,
    scaffold_test_split,
    build_parent_scaffold_df,
    scaffold_kfold_split,
    train_one_fold,
    active_feature_dim,
)


# ═════════════════════════════════════════════════════════════════
# The variant being tuned: the reference full model
# ═════════════════════════════════════════════════════════════════
VARIANT = VariantConfig(
    name="full_model_diversity_reranker",
    description=("Reference full model: GATv2 encoder, 19-d atom features, "
                 "learned reranker with the diversity-aware ranking objective"),
    gnn="gatv2",
    feature_mode="full",
    train_som=True,
    train_rxn=True,
    use_reranker=True,
    use_diversity=True,
    generate_metabolites=True,
    selection_metric="joint",
)


# ═════════════════════════════════════════════════════════════════
# SEARCH SPACE
# ═════════════════════════════════════════════════════════════════

SEARCH_SPACE: List[Tuple[str, Tuple]] = [
    ("layers",      (2, 3, 4, 5)),
    ("heads",       (2, 3, 4, 5, 6, 7, 8)),
    ("hidden_som",  (64, 128, 256)),
    ("hidden_rxn",  (64, 128, 256)),
    ("hidden_rer",  (64, 128, 256)),
    ("dropout",     (0.1, 0.2, 0.3)),
    ("drop_edge_p", (0.1, 0.2, 0.3)),
]
PARAM_NAMES = [n for n, _ in SEARCH_SPACE]
INT_PARAMS = {"layers", "heads", "hidden_som", "hidden_rxn", "hidden_rer"}

# CLI flag name for each parameter (argparse dest -> command-line spelling)
CLI_FLAG = {
    "layers":      "--layers",
    "heads":       "--heads",
    "hidden_som":  "--hidden-som",
    "hidden_rxn":  "--hidden-rxn",
    "hidden_rer":  "--hidden-rer",
    "dropout":     "--dropout",
    "drop_edge_p": "--drop-edge-p",
}

PENALTY = 1.0   # returned to the optimiser when a trial fails (objective is -joint in [-1, 0])


def gpyopt_domain() -> List[Dict]:
    """GPyOpt domain specification. All variables are ordinal discrete."""
    return [{"name": name, "type": "discrete", "domain": tuple(values)}
            for name, values in SEARCH_SPACE]


def grid_size() -> int:
    n = 1
    for _, v in SEARCH_SPACE:
        n *= len(v)
    return n


def snap(value: float, allowed: Tuple) -> Any:
    """Snap a continuous proposal onto the nearest allowed grid value."""
    arr = np.asarray(allowed, dtype=float)
    return allowed[int(np.argmin(np.abs(arr - float(value))))]


def decode(x_row) -> Dict[str, Any]:
    """Map one row of the GPyOpt design matrix to a hyperparameter dict."""
    cfg: Dict[str, Any] = {}
    for i, (name, allowed) in enumerate(SEARCH_SPACE):
        v = snap(float(x_row[i]), allowed)
        cfg[name] = int(v) if name in INT_PARAMS else float(round(float(v), 6))
    return cfg


def encode(cfg: Dict[str, Any]) -> List[float]:
    """Inverse of decode(), for seeding the GP from a resumed trial history."""
    return [float(cfg[name]) for name in PARAM_NAMES]


def config_key(cfg: Dict[str, Any]) -> Tuple:
    """Hashable identity of a configuration, used to memoise repeated proposals."""
    return tuple((name, round(float(cfg[name]), 6)) for name in PARAM_NAMES)


def config_str(cfg: Dict[str, Any]) -> str:
    return "  ".join(f"{k}={cfg[k]}" for k in PARAM_NAMES)


def cli_args_for(cfg: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for name in PARAM_NAMES:
        out += [CLI_FLAG[name], str(cfg[name])]
    return out


# ═════════════════════════════════════════════════════════════════
# DATA, PREPARED ONCE AND REUSED BY EVERY TRIAL
# ═════════════════════════════════════════════════════════════════
class FoldData:

    def __init__(self, args, log_fp: str):
        set_feature_mode(VARIANT.feature_mode)
        enable_reproducibility(args.seed)

        print("Loading dataset ...")
        self.sg, self.rg, self.le, self.df = load_dataset(args.csv)
        self.num_classes = len(self.le.classes_)

        pre_idx, tst_idx, holdout_info, _ = scaffold_test_split(self.df, 0.20, args.seed)
        self.pre_sg = [self.sg[i] for i in pre_idx]
        self.pre_rg = [self.rg[i] for i in pre_idx]
        self.pre_df = self.df.iloc[pre_idx].reset_index(drop=True)
        self.n_test = len(tst_idx)          # reported only; never evaluated here

        parent_df = build_parent_scaffold_df(self.pre_df).reset_index(drop=True)
        parent_df["parent_index"] = np.arange(len(parent_df), dtype=int)
        parent_to_rows = (self.pre_df
                          .assign(parent_id=self.pre_df["Predecessor_SMILES"]
                                  .astype(str).str.strip())
                          .groupby("parent_id").indices)

        splits = scaffold_kfold_split(parent_df, args.hp_kfolds, args.seed)
        self.folds: List[Tuple[np.ndarray, np.ndarray]] = []
        for tri_p, vli_p in splits:
            tr_ids = parent_df.iloc[tri_p]["parent_id"].tolist()
            vl_ids = parent_df.iloc[vli_p]["parent_id"].tolist()
            tri = np.array(sorted(r for pid in tr_ids for r in parent_to_rows[pid]), dtype=int)
            vli = np.array(sorted(r for pid in vl_ids for r in parent_to_rows[pid]), dtype=int)
            self.folds.append((tri, vli))

        msg = (f"  molecules={len(self.sg)}  development={len(self.pre_sg)}  "
               f"test={self.n_test} (untouched)  classes={self.num_classes}  "
               f"atom_features={active_feature_dim()}\n"
               f"  search CV: {args.hp_kfolds} scaffold folds, sizes "
               + ", ".join(f"{len(v)}" for _, v in self.folds)
               + f"\n  holdout split method={holdout_info['method']} "
                 f"test_scaffolds={holdout_info['n_test_scaffolds']}")
        print(msg)
        write_log(log_fp, msg)

    def fold(self, k: int):
        tri, vli = self.folds[k]
        return ([self.pre_sg[i] for i in tri], [self.pre_sg[i] for i in vli],
                [self.pre_rg[i] for i in tri], [self.pre_rg[i] for i in vli],
                self.pre_df.iloc[tri].reset_index(drop=True))


# ═════════════════════════════════════════════════════════════════
# ONE TRIAL = CROSS-VALIDATION AT ONE HYPERPARAMETER SETTING
# ═════════════════════════════════════════════════════════════════
def trial_args(base_args, cfg: Dict[str, Any]):
    """Base CLI arguments with the trial's hyperparameters and search budget."""
    a = copy.deepcopy(base_args)
    for name in PARAM_NAMES:
        setattr(a, name, cfg[name])
    a.epochs = int(base_args.hp_epochs)
    a.patience = int(base_args.hp_patience)
    a.kfolds = int(base_args.hp_kfolds)
    a.save_images = 0                 # never render images during a search
    return a


def prune_trial_dir(path: str):
    """Delete checkpoints and scalers, keep logs / CSV / JSON for the record."""
    for root, _dirs, files in os.walk(path):
        for fn in files:
            if fn.endswith((".pt", ".pkl", ".png")):
                try:
                    os.remove(os.path.join(root, fn))
                except OSError:
                    pass


def run_trial(trial_id: int, cfg: Dict[str, Any], data: FoldData,
              base_args, device, search_dir: str, log_fp: str) -> Dict[str, Any]:
    t0 = time.time()
    tdir = os.path.join(search_dir, f"trial_{trial_id:04d}")
    ensure_dir(tdir)
    a = trial_args(base_args, cfg)

    header = (f"\n{'=' * 78}\n  TRIAL {trial_id:04d}  |  {config_str(cfg)}\n"
              f"  budget: {a.kfolds} folds x {a.epochs} epochs (patience {a.patience})\n"
              f"{'=' * 78}")
    print(header)
    write_log(log_fp, header)

    scores: List[float] = []
    epochs: List[int] = []
    status = "ok"

    try:
        for k in range(len(data.folds)):
            tr_sg, vl_sg, tr_rg, vl_rg, tr_df = data.fold(k)
            res = train_one_fold(VARIANT, k + 1, tr_sg, vl_sg, tr_rg, vl_rg,
                                 tr_df, a, device, tdir, data.le, data.num_classes)
            scores.append(float(res["best_selection_score"]))
            epochs.append(int(res["best_epoch"]))
            # release the cached state dicts immediately: with many trials these
            # dominate host memory
            for key in ("som_state", "rxn_state", "rer_state",
                        "scaler_som", "scaler_rxn"):
                res[key] = None
            del res
            if device.type == "cuda":
                torch.cuda.empty_cache()

    except (torch.cuda.OutOfMemoryError if hasattr(torch.cuda, "OutOfMemoryError")
            else RuntimeError) as exc:                                  # noqa: B030
        status = f"oom_or_runtime: {exc}"
        print(f"  [trial {trial_id:04d}] FAILED ({status})")
        write_log(log_fp, f"  FAILED {status}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:                                            # noqa: BLE001
        status = f"error: {exc}"
        print(f"  [trial {trial_id:04d}] FAILED ({status})")
        write_log(log_fp, "  FAILED\n" + traceback.format_exc())
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if scores:
        mean = float(np.mean(scores))
        sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        if base_args.objective == "mean_minus_sd":
            score = mean - float(base_args.stability_lambda) * sd
        else:
            score = mean
        objective = -score
    else:
        mean = sd = score = float("nan")
        objective = PENALTY

    rec: Dict[str, Any] = dict(cfg)
    rec.update({
        "trial": trial_id,
        "status": status,
        "n_folds_completed": len(scores),
        "fold_scores": json.dumps([round(s, 6) for s in scores]),
        "fold_best_epochs": json.dumps(epochs),
        "mean_joint": round(mean, 6) if scores else "",
        "sd_joint": round(sd, 6) if scores else "",
        "objective_score": round(score, 6) if scores else "",
        "objective_value": round(float(objective), 6),
        "mean_best_epoch": int(round(float(np.mean(epochs)))) if epochs else "",
        "seconds": round(time.time() - t0, 1),
    })

    line = (f"  TRIAL {trial_id:04d} done in {rec['seconds']}s  "
            f"mean_joint={rec['mean_joint']}  sd={rec['sd_joint']}  "
            f"objective_score={rec['objective_score']}  status={status}")
    print(line)
    write_log(log_fp, line)

    if not base_args.keep_checkpoints:
        prune_trial_dir(tdir)
    return rec


# ═════════════════════════════════════════════════════════════════
# TRIAL HISTORY (CSV, written after every trial so a crash loses nothing)
# ═════════════════════════════════════════════════════════════════
CSV_FIELDS = (PARAM_NAMES +
              ["trial", "status", "n_folds_completed", "fold_scores",
               "fold_best_epochs", "mean_joint", "sd_joint", "objective_score",
               "objective_value", "mean_best_epoch", "seconds"])


def append_trial(path: str, rec: Dict[str, Any]):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow({k: rec.get(k, "") for k in CSV_FIELDS})


def load_trials(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                cfg = {n: (int(float(row[n])) if n in INT_PARAMS else float(row[n]))
                       for n in PARAM_NAMES}
            except (KeyError, ValueError):
                continue
            rec = dict(row)
            rec.update(cfg)
            rec["trial"] = int(float(row.get("trial", 0) or 0))
            try:
                rec["objective_value"] = float(row.get("objective_value", PENALTY))
            except ValueError:
                rec["objective_value"] = PENALTY
            out.append(rec)
    return out


# ═════════════════════════════════════════════════════════════════
# ARGUMENTS
# ═════════════════════════════════════════════════════════════════
def build_parser():
    p = build_argparser(VARIANT)
    p.description = ("GPyOpt Bayesian hyperparameter optimisation for the "
                     "Metabolite-GEN reference full model")

    g = p.add_argument_group("hyperparameter search")
    g.add_argument("--backend", type=str, default="gpyopt",
                   choices=["gpyopt", "random"],
                   help="GP-EI Bayesian optimisation, or random search over the "
                        "same grid if GPyOpt cannot be installed")
    g.add_argument("--n-init", type=int, default=10,
                   help="Random configurations evaluated before the GP takes over")
    g.add_argument("--hp-max-iter", type=int, default=40,
                   help="Bayesian optimisation iterations after the initial design")
    g.add_argument("--hp-kfolds", type=int, default=3,
                   help="CV folds per trial. Lower is cheaper; set 5 to match the "
                        "protocol used for the reported results")
    g.add_argument("--hp-epochs", type=int, default=60,
                   help="Epoch cap per fold during the search")
    g.add_argument("--hp-patience", type=int, default=12,
                   help="Early-stopping patience per fold during the search")
    g.add_argument("--objective", type=str, default="mean",
                   choices=["mean", "mean_minus_sd"],
                   help="Maximise the mean fold joint metric, or penalise "
                        "fold-to-fold instability")
    g.add_argument("--stability-lambda", type=float, default=1.0,
                   help="Weight of the SD penalty when --objective mean_minus_sd")
    g.add_argument("--acquisition", type=str, default="EI",
                   choices=["EI", "MPI", "LCB"], help="GPyOpt acquisition function")
    g.add_argument("--acquisition-jitter", type=float, default=0.01,
                   help="Exploration parameter of the acquisition function")
    g.add_argument("--keep-checkpoints", type=int, default=0,
                   help="Keep per-trial .pt/.pkl/.png files (large); logs and "
                        "CSV/JSON are always kept")
    g.add_argument("--resume", type=int, default=0,
                   help="Reload hp_trials.csv, seed the GP with it, and continue")
    g.add_argument("--dry-run", type=int, default=0,
                   help="Print the search plan and exit without training")
    g.add_argument("--run-final", type=int, default=0,
                   help="After the search, immediately retrain with the winning "
                        "configuration using the standard two-stage protocol")
    return p


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════
def main():
    args = build_parser().parse_args()
    VARIANT.check()

    search_dir = os.path.join(args.outdir, "hp_search")
    ensure_dir(args.outdir)
    ensure_dir(search_dir)
    log_fp = os.path.join(search_dir, "hp_search_log.txt")
    csv_fp = os.path.join(search_dir, "hp_trials.csv")

    device = torch.device(args.device if torch.cuda.is_available()
                          and args.device.startswith("cuda") else "cpu")

    n_trials = int(args.n_init) + int(args.hp_max_iter)
    plan = [
        "=" * 78,
        "  GPyOpt HYPERPARAMETER OPTIMISATION — Metabolite-GEN reference full model",
        "=" * 78,
        f"  backend        : {args.backend}",
        f"  device         : {device}",
        f"  search space   : {grid_size():,} configurations over {len(SEARCH_SPACE)} parameters",
    ]
    for name, values in SEARCH_SPACE:
        plan.append(f"      {name:12s} {list(values)}")
    plan += [
        "  objective      : maximise "
        + ("mean(joint)" if args.objective == "mean"
           else f"mean(joint) - {args.stability_lambda} * SD(joint)"),
        f"  trial budget   : {args.n_init} initial + {args.hp_max_iter} BO = {n_trials} trials",
        f"  cost per trial : {args.hp_kfolds} folds x up to {args.hp_epochs} epochs "
        f"(patience {args.hp_patience})",
        f"  total folds    : up to {n_trials * args.hp_kfolds}",
        f"  outputs        : {search_dir}",
        "  the held-out test set is NOT used anywhere in this script",
        "=" * 78,
    ]
    text = "\n".join(plan)
    print(text)
    write_log(log_fp, text)

    if int(args.dry_run):
        print("\nDry run: no training performed. Re-run without --dry-run 1 to start.")
        return

    # ── optimiser backend ─────────────────────────────────────────
    BayesianOptimization = None
    if args.backend == "gpyopt":
        try:
            from GPyOpt.methods import BayesianOptimization  # noqa: N806
        except Exception as exc:                             # noqa: BLE001
            raise SystemExit(
                "\nGPyOpt could not be imported: "
                f"{exc}\n\n"
                "GPyOpt is unmaintained and does not build against NumPy >= 2.\n"
                "Try:\n"
                '    pip install "numpy<2" "scipy<1.14" GPy GPyOpt matplotlib\n\n'
                "Or re-run this script with --backend random to use random search\n"
                "over the same grid; all outputs and downstream commands are identical."
            ) from exc

    data = FoldData(args, log_fp)

    # ── state shared by the objective closure ─────────────────────
    history: List[Dict[str, Any]] = []
    cache: Dict[Tuple, float] = {}
    counter = {"n": 0}

    if int(args.resume):
        history = load_trials(csv_fp)
        for rec in history:
            cfg = {n: rec[n] for n in PARAM_NAMES}
            cache[config_key(cfg)] = float(rec["objective_value"])
        counter["n"] = max([int(r["trial"]) for r in history], default=0)
        msg = f"Resumed {len(history)} completed trials from {csv_fp}."
        print(msg); write_log(log_fp, msg)

    def evaluate(cfg: Dict[str, Any]) -> float:
        key = config_key(cfg)
        if key in cache:
            print(f"  (cached) {config_str(cfg)} -> objective {cache[key]:.6f}")
            return cache[key]
        counter["n"] += 1
        rec = run_trial(counter["n"], cfg, data, args, device, search_dir, log_fp)
        append_trial(csv_fp, rec)
        history.append(rec)
        cache[key] = float(rec["objective_value"])
        return cache[key]

    def objective(X) -> np.ndarray:
        X = np.atleast_2d(X)
        return np.array([[evaluate(decode(X[i]))] for i in range(X.shape[0])])

    # ── run the search ────────────────────────────────────────────
    rng = np.random.RandomState(int(args.seed))
    t_start = time.time()

    if args.backend == "random":
        for _ in range(n_trials - len(history)):
            cfg = {name: (int(values[rng.randint(len(values))]) if name in INT_PARAMS
                          else float(values[rng.randint(len(values))]))
                   for name, values in SEARCH_SPACE}
            evaluate(cfg)
        optimiser = None
    else:
        seed_X = seed_Y = None
        if history:
            seed_X = np.array([encode({n: r[n] for n in PARAM_NAMES}) for r in history],
                              dtype=float)
            seed_Y = np.array([[float(r["objective_value"])] for r in history], dtype=float)

        optimiser = BayesianOptimization(
            f=objective,
            domain=gpyopt_domain(),
            model_type="GP",
            acquisition_type=args.acquisition,
            acquisition_jitter=float(args.acquisition_jitter),
            initial_design_numdata=max(0, int(args.n_init) - len(history)),
            initial_design_type="random",
            normalize_Y=True,
            exact_feval=False,      # CV scores are noisy across folds
            maximize=False,         # we return -score
            X=seed_X, Y=seed_Y,
            verbosity=True,
        )
        optimiser.run_optimization(
            max_iter=int(args.hp_max_iter),
            report_file=os.path.join(search_dir, "gpyopt_report.txt"),
            evaluations_file=os.path.join(search_dir, "gpyopt_evaluations.txt"),
            models_file=os.path.join(search_dir, "gpyopt_models.txt"),
            verbosity=True,
        )
        try:
            import matplotlib
            matplotlib.use("Agg")
            optimiser.plot_convergence(
                filename=os.path.join(search_dir, "gpyopt_convergence.png"))
        except Exception as exc:                                        # noqa: BLE001
            print(f"  [warn] convergence plot skipped: {exc}")

    elapsed = time.time() - t_start

    # ── report ────────────────────────────────────────────────────
    valid = [r for r in history if r.get("status") == "ok" and r.get("mean_joint") not in ("", None)]
    if not valid:
        raise SystemExit("No trial completed successfully — check hp_search_log.txt.")

    best = min(valid, key=lambda r: float(r["objective_value"]))
    best_cfg = {n: best[n] for n in PARAM_NAMES}

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(search_dir, "hp_trials_final.csv"), index=False)

    # Marginal effect of each parameter, for the sensitivity figure (R1 #6)
    rows = []
    vdf = pd.DataFrame(valid)
    vdf["mean_joint"] = pd.to_numeric(vdf["mean_joint"], errors="coerce")
    for name, values in SEARCH_SPACE:
        for v in values:
            sub = vdf[pd.to_numeric(vdf[name], errors="coerce") == float(v)]
            if len(sub):
                rows.append({"parameter": name, "value": v, "n_trials": len(sub),
                             "mean_joint_mean": round(float(sub["mean_joint"].mean()), 6),
                             "mean_joint_sd": (round(float(sub["mean_joint"].std(ddof=1)), 6)
                                               if len(sub) > 1 else 0.0),
                             "mean_joint_max": round(float(sub["mean_joint"].max()), 6)})
    pd.DataFrame(rows).to_csv(
        os.path.join(search_dir, "hp_marginal_effects.csv"), index=False)

    cli = cli_args_for(best_cfg)
    with open(os.path.join(search_dir, "best_cli_args.txt"), "w", encoding="utf-8") as fh:
        fh.write(" ".join(cli) + "\n")
    with open(os.path.join(search_dir, "best_hyperparameters.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "variant": VARIANT.name,
            "best_hyperparameters": best_cfg,
            "cli_args": cli,
            "search_objective": args.objective,
            "stability_lambda": args.stability_lambda,
            "search_mean_joint": best.get("mean_joint"),
            "search_sd_joint": best.get("sd_joint"),
            "search_fold_scores": best.get("fold_scores"),
            "search_fold_best_epochs": best.get("fold_best_epochs"),
            "suggested_final_epochs": best.get("mean_best_epoch"),
            "search_budget": {"backend": args.backend, "n_init": args.n_init,
                              "max_iter": args.hp_max_iter, "kfolds": args.hp_kfolds,
                              "epochs": args.hp_epochs, "patience": args.hp_patience},
            "n_trials_completed": len(valid),
            "search_space": {n: list(v) for n, v in SEARCH_SPACE},
            "grid_size": grid_size(),
            "wall_clock_seconds": round(elapsed, 1),
            "note": ("Selected by cross-validation on the development set only. "
                     "The held-out test set was not used at any point in this search."),
        }, fh, indent=2)

    summary = [
        "", "=" * 78,
        "  BEST CONFIGURATION",
        "=" * 78,
        f"  {config_str(best_cfg)}",
        f"  trial          : {best['trial']}",
        f"  mean joint     : {best['mean_joint']}  (SD {best['sd_joint']} across "
        f"{args.hp_kfolds} folds)",
        f"  fold scores    : {best['fold_scores']}",
        f"  mean best epoch: {best['mean_best_epoch']}",
        f"  trials run     : {len(valid)} successful / {len(history)} total "
        f"in {elapsed / 3600:.2f} h",
        "",
        "  Retrain the final model with:",
        "",
        "    python variant_full_model_diversity_reranker_v1.py \\",
        f"        --csv {args.csv} \\",
        f"        --outdir {os.path.join(args.outdir, 'full_model_tuned')} \\",
        f"        {' '.join(cli)} \\",
        f"        --epochs {args.epochs} --patience {args.patience} --device {args.device}",
        "",
        "  Reminder: --hp-epochs capped the search at "
        f"{args.hp_epochs} epochs, so use the full --epochs budget for the final",
        "  run and let the standard protocol derive --final-epochs from its own CV.",
        "=" * 78,
    ]
    text = "\n".join(summary)
    print(text)
    write_log(log_fp, text)

    # ── optional: retrain immediately with the winning configuration ──
    if int(args.run_final):
        final_out = os.path.join(args.outdir, "full_model_tuned")
        argv = (["--csv", args.csv, "--outdir", final_out] + cli +
                ["--epochs", str(args.epochs),
                 "--patience", str(args.patience),
                 "--kfolds", str(args.kfolds),
                 "--device", args.device,
                 "--seed", str(args.seed),
                 "--amp", str(args.amp),
                 "--num-workers", str(args.num_workers),
                 "--lr", str(args.lr),
                 "--lr-min", str(args.lr_min),
                 "--lr-rer", str(args.lr_rer),
                 "--alpha", str(args.alpha),
                 "--gamma", str(args.gamma),
                 "--focal-gamma-rxn", str(args.focal_gamma_rxn),
                 "--diversity-weight", str(args.diversity_weight),
                 "--aug-threshold", str(args.aug_threshold),
                 "--aug-factor", str(args.aug_factor),
                 "--top-som", str(args.top_som),
                 "--top-rxn", str(args.top_rxn),
                 "--tanimoto-threshold", str(args.tanimoto_threshold),
                 "--final-epochs-rule", str(args.final_epochs_rule),
                 "--save-images", str(args.save_images)])
        banner = ("\n" + "=" * 78 +
                  "\n  RETRAINING WITH THE WINNING CONFIGURATION\n" + "=" * 78)
        print(banner); write_log(log_fp, banner)
        run_pipeline(VARIANT, argv)

    print(f"\nSearch artefacts written to: {search_dir}")


if __name__ == "__main__":
    main()
