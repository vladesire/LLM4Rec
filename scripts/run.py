#!/usr/bin/env python3
"""Reproducible entry point; never uploads or submits anything."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core import ap_at_5
from data_pipeline import dump, human, prepare, sha256
from models import fit_score, reciprocal_rank_fusion


def evaluate(result, relevant, users, warm_users):
    ap = [ap_at_5(relevant.get(u, set()), r) for u, r in zip(users, result["predictions"])]
    nonempty = [i for i, u in enumerate(users) if relevant.get(u)]
    warm = [i for i, u in enumerate(users) if u in warm_users]
    cold = [i for i, u in enumerate(users) if u not in warm_users]
    return {"map_at_5": float(np.mean(ap)), "evaluation_users": len(users),
            "nonempty_relevance_users": len(nonempty), "empty_relevance_users_included_as_zero": len(users) - len(nonempty),
            "map_at_5_nonempty_secondary": float(np.mean([ap[i] for i in nonempty])) if nonempty else None,
            "warm_users": len(warm), "cold_users": len(cold),
            "warm_map_at_5": float(np.mean([ap[i] for i in warm])) if warm else None,
            "cold_map_at_5": float(np.mean([ap[i] for i in cold])) if cold else None,
            "fit_seconds": result["fit_seconds"], "inference_seconds": result["inference_seconds"],
            **result["details"]}, ap


def validate_submission(frame, template, candidates):
    assert list(frame.columns) == list(template.columns), "Template column mismatch"
    assert len(frame) == len(template) and frame.user_id.tolist() == template.user_id.tolist()
    assert not frame.isna().any().any()
    eligible = set(candidates)
    for value in frame.tv_show_id:
        tokens = value.split(" ")
        assert len(tokens) == 5 and all(t.isdigit() and t == str(int(t)) for t in tokens)
        ids = [int(t) for t in tokens]
        assert len(set(ids)) == 5 and all(i > 0 and i in eligible for i in ids)
    return {"passed": True, "rows": len(frame), "columns": list(frame.columns),
            "exact_template_user_order": True, "five_distinct_nonzero_integer_ids": True,
            "eligible_items_only": True, "no_nulls": True, "single_spaces": True, "no_csv_index": True}


def save_examples(data, results, ap_values, relevant, output, selected):
    train, val = data["train"], data["val"]
    sizes = train.groupby("u").item.nunique()
    eligible = [u for u in data["evaluation_users"] if relevant.get(u) and sizes.get(u, 0) > 0]
    ordered = sorted(eligible, key=lambda u: (sizes.get(u, 0), data["users"][u]))
    # Contrasting history lengths with nonempty held-out viewing.
    chosen = [ordered[max(0, len(ordered) // 10)], ordered[min(len(ordered) - 1, len(ordered) * 9 // 10)]]
    meta = data["metadata"].drop_duplicates("item", keep="first").set_index("item")
    candidate_positions = {int(item): i for i, item in enumerate(data["validation_candidates"])}

    def describe(item):
        if item not in meta.index:
            return {"item": int(item), "title": "[No pre-test metadata]", "category": "[Unknown]"}
        row = meta.loc[item]
        return {"item": int(item), "title": str(row.tv_show_title), "category": str(row.tv_show_category)}

    examples, histories = [], []
    for u in chosen:
        pos = data["evaluation_users"].index(u)
        ex = {"user_id": data["users"][u], "training_distinct_items": int(sizes.get(u, 0)),
              "held_out_distinct_items": len(relevant.get(u, set())), "methods": {}}
        for phase, interaction in [("fitting", train), ("held_out", val)]:
            for row in interaction.loc[interaction.u == u].sort_values(["seconds", "item"], ascending=[False, True]).itertuples():
                histories.append({"user_id": data["users"][u], "phase": phase, **describe(row.item), "watched_seconds": int(row.seconds)})
        for name, result in results.items():
            recommendations = []
            for rank, item in enumerate(result["predictions"][pos], 1):
                ix = candidate_positions[item]
                raw = float(result["scores"][pos, ix])
                fallback = bool(result["fallback"][pos, rank - 1])
                recommendations.append({"rank": rank, **describe(item), "method_score": raw if np.isfinite(raw) else None,
                                        "eligible_popularity": float(result["popularity"][ix]), "popularity_backfill": fallback,
                                        "held_out_relevant": item in relevant.get(u, set())})
            ex["methods"][name] = {"ap_at_5": ap_values[name][pos], "recommendations": recommendations}
        examples.append(ex)
    dump(output / "two_user_examples.json", examples)
    pd.DataFrame(histories).to_csv(output / "two_user_histories.csv", index=False)
    lines = ["# Same two users for every method", "", "Examples use validation viewing. Test-period true viewing is unavailable.",
             "Titles/categories are for display from pre-test schedule metadata; models use only their permitted fitting-prefix metadata.", ""]
    for ex in examples:
        lines += [f"## User {ex['user_id']}", "", f"Training distinct items: {ex['training_distinct_items']}; held-out distinct items: {ex['held_out_distinct_items']}.", "",
                  "| Method | Rank | ID | Title | Category | Method score | Backfill | Relevant |", "|---|---:|---:|---|---|---:|---|---|"]
        for name, value in ex["methods"].items():
            for rec in value["recommendations"]:
                title = rec['title'].replace('|', '/')
                lines.append(f"| {name} | {rec['rank']} | {rec['item']} | {title} | {rec['category']} | {rec['method_score']} | {rec['popularity_backfill']} | {rec['held_out_relevant']} |")
        lines.append("")
    (output / "two_user_examples.md").write_text("\n".join(lines))
    # Choose a real evaluated row with a hit, preferably one of the two examples.
    positions = [data["evaluation_users"].index(u) for u in chosen]
    positions += list(np.argsort(-np.asarray(ap_values[selected])))
    pos = next((i for i in positions if ap_values[selected][i] > 0), positions[0])
    u = data["evaluation_users"][pos]
    rel = relevant.get(u, set())
    hits, total, ranks = 0, 0.0, []
    for rank, item in enumerate(results[selected]["predictions"][pos], 1):
        hit = item in rel
        hits += int(hit)
        contribution = hits / rank if hit else 0
        total += contribution
        ranks.append({"rank": rank, **describe(item), "relevant": hit, "hits_so_far": hits,
                      "precision_at_rank": hits / rank, "ap_numerator_contribution": contribution})
    worked = {"user_id": data["users"][u], "method": selected, "distinct_relevant_ids": sorted(rel),
              "number_distinct_relevant": len(rel), "denominator": min(5, len(rel)), "rank_calculation": ranks,
              "numerator": total, "ap_at_5": ap_values[selected][pos], "student_personally_verified": False}
    dump(output / "real_user_ap_example.json", worked)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--submission-dir", default="submission")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--hybrid", action="store_true", help="Evaluate one equal-weight content/ALS fusion after individual methods")
    args = parser.parse_args()
    start = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    assert config["allow_repeats"], "This implementation documents and uses repeat-allowed recommendations"
    out, sub = Path(args.output_dir), Path(args.submission_dir)
    out.mkdir(parents=True, exist_ok=True)
    sub.mkdir(parents=True, exist_ok=True)
    data = prepare(args.data_dir, args.cache_dir, out, config, args.rebuild_data)
    relevant = {int(u): set(g.item.tolist()) for u, g in data["val"].groupby("u")}
    warm_users = set(data["train"].u)
    candidates = data["validation_candidates"]
    unreachable = sum(len(items - set(candidates)) for u, items in relevant.items() if u in set(data["evaluation_users"]))
    assert unreachable == 0
    results, metrics, ap_values = {}, {}, {}
    for method in ["popularity", "content", "als"]:
        result = fit_score(method, data["train"], data["metadata"], data["cutoff"], len(data["users"]), data["evaluation_users"], candidates, config)
        metric, aps = evaluate(result, relevant, data["evaluation_users"], warm_users)
        results[method], metrics[method], ap_values[method] = result, metric, aps
        dump(out / "metrics_checkpoint.json", metrics)
        print(f"CHECKPOINT {method}: MAP@5={metric['map_at_5']:.8f}; fit={metric['fit_seconds']:.3f}s; inference={metric['inference_seconds']:.3f}s", flush=True)
    if args.hybrid:
        result = reciprocal_rank_fusion({k: results[k] for k in ["content", "als"]}, candidates)
        metric, aps = evaluate(result, relevant, data["evaluation_users"], warm_users)
        results["hybrid"], metrics["hybrid"], ap_values["hybrid"] = result, metric, aps
        print(f"CHECKPOINT hybrid: MAP@5={metric['map_at_5']:.8f}", flush=True)
    selected = max(metrics, key=lambda name: metrics[name]["map_at_5"])
    record = {"primary_population": "All official template users, including empty held-out relevance as AP=0",
              "primary_metric": "AP@5=sum of precision at relevant recommended ranks / min(5, distinct held-out relevant items); mean over declared population",
              "unreachable_distinct_user_item_labels": unreachable, "candidate_items": len(candidates),
              "fitting_users": int(data["train"].u.nunique()), "evaluation_users": len(data["evaluation_users"]),
              "fitting_interactions": len(data["train"]), "held_out_interactions_all_log_users": len(data["val"]),
              "held_out_interactions_template_users": sum(len(relevant.get(u, set())) for u in data["evaluation_users"]),
              "models": metrics, "selected_method": selected, "config": config,
              "selection_note": "One documented temporal hold-out is used for comparison and selection; no unbiased second evaluation or Kaggle score tuning.",
              "optional_fusion_evaluated": args.hybrid}
    dump(out / "metrics.json", record)
    table = pd.DataFrame([{"method": name, "MAP@5": r["map_at_5"], "fit_seconds": r["fit_seconds"],
                           "inference_seconds": r["inference_seconds"], "method_slot_coverage": r["method_slots_fraction"],
                           "cold_users": r["cold_users"], "distinct_recommended_items": r["distinct_recommended_items"]} for name, r in metrics.items()])
    table.to_csv(out / "comparison.csv", index=False)
    md = ["| Method | MAP@5 | Fit seconds | Inference seconds | Method slot coverage | Cold users |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in metrics.items():
        md.append(f"| {name} | {r['map_at_5']:.6f} | {r['fit_seconds']:.3f} | {r['inference_seconds']:.3f} | {r['method_slots_fraction']:.3f} | {r['cold_users']} |")
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    save_examples(data, results, ap_values, relevant, out, selected)
    # Save exact held-out predictions and per-user AP for deterministic reruns.
    evidence = {name: {"predictions": r["predictions"], "ap_at_5": ap_values[name]} for name, r in results.items()}
    dump(out / "validation_predictions.json", evidence)
    print("Selection complete:", selected, "Production refit starts on all eligible pre-test history", flush=True)
    t = time.perf_counter()
    if selected == "hybrid":
        components = {name: fit_score(name, data["production"], data["metadata"], data["test_start"], len(data["users"]),
                                     data["evaluation_users"], data["test_candidates"], config) for name in ["content", "als"]}
        production = reciprocal_rank_fusion(components, data["test_candidates"])
    else:
        production = fit_score(selected, data["production"], data["metadata"], data["test_start"], len(data["users"]),
                               data["evaluation_users"], data["test_candidates"], config)
    frame = data["template"].copy()
    frame["tv_show_id"] = [" ".join(map(str, row)) for row in production["predictions"]]
    verification = validate_submission(frame, data["template"], data["test_candidates"])
    path = sub / "final_submission.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    reread = pd.read_csv(path, dtype=str, keep_default_na=False)
    validate_submission(reread, data["template"], data["test_candidates"])
    checksum = sha256(path)
    (sub / "final_submission.csv.sha256").write_text(checksum + "  final_submission.csv\n")
    verification.update({"sha256": checksum, "selected_method": selected, "file": str(path.resolve()),
                         "production_refit_seconds_including_inference": time.perf_counter() - t,
                         "production_model_details": production["details"], "production_fit_seconds": production["fit_seconds"],
                         "production_inference_seconds": production["inference_seconds"]})
    dump(out / "submission_validation.json", verification)
    environment = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                   "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "implicit", "reportlab", "pymupdf"]},
                   "seed": config["seed"], "config": config, "command_argv": sys.argv,
                   "elapsed_entry_point_seconds": time.perf_counter() - start,
                   "script_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                   "thread_environment": {k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]}}
    dump(out / "environment.json", environment)
    print("VALIDATED", len(frame), "rows; SHA-256", checksum, flush=True)


if __name__ == "__main__":
    main()
