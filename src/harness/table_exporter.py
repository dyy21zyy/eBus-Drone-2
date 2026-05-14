from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from src.utils.metrics import REQUIRED_PAPER_METRICS

DISPLAY = {
    "total_cost": ("Total weighted cost", "cost"),
    "onboard_passenger_delay": ("Onboard passenger delay", "min"),
    "parcel_lateness": ("Parcel lateness", "min"),
    "late_delivery_count": ("Late deliveries", "count"),
    "undelivered_parcel_count": ("Undelivered parcels", "count"),
    "total_energy_consumption": ("Energy consumption", "kWh"),
    "station_power_overload_amount": ("Power overload amount", "kW"),
    "station_power_overload_duration": ("Power overload duration", "min"),
    "locker_overflow_amount": ("Locker overflow amount", "kg"),
    "locker_overflow_duration": ("Locker overflow duration", "min"),
    "minimum_bus_battery": ("Minimum bus battery", "kWh"),
    "runtime_sec": ("Runtime", "sec"),
    "evaluation_time_sec": ("Evaluation time", "sec"),
    "training_time_sec": ("Training time", "sec"),
}

EXPERIMENT_DEFAULT_METRICS = {
    "overall": [
        "total_cost", "onboard_passenger_delay", "parcel_lateness", "late_delivery_count",
        "undelivered_parcel_count", "total_energy_consumption", "station_power_overload_amount",
        "station_power_overload_duration", "locker_overflow_amount", "locker_overflow_duration",
        "minimum_bus_battery", "runtime_sec", "evaluation_time_sec"
    ],
    "ablation": ["total_cost", "onboard_passenger_delay", "parcel_lateness", "total_energy_consumption", "minimum_bus_battery"],
    "scalability": ["num_customers", "num_stations", "num_bus_trips", "total_cost", "runtime_sec", "training_time_sec", "evaluation_time_sec"],
    "sensitivity": ["total_cost", "onboard_passenger_delay", "parcel_lateness", "total_energy_consumption"],
}


@dataclass
class ExportResult:
    raw_csv: str
    aggregated_csv: str
    tex_path: str
    meta_json: str


def _float(v: str) -> float:
    return float(v)


def _format(mean_v: float, std_v: float) -> str:
    return f"{mean_v:.4f} ± {std_v:.4f}"


def _required_columns(experiment: str) -> list[str]:
    required = ["method", "seed"]
    if experiment == "sensitivity":
        required.append("value")
    return required


def export_tables(output_root: Path, experiment: str, instance: str, method_order: list[str], metrics_subset: list[str] | None = None, config_snapshot: dict | None = None) -> ExportResult:
    summary_csv = output_root / "results" / experiment / instance / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing results CSV: {summary_csv}")
    rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError("Empty input results: summary.csv has no rows.")

    req_cols = _required_columns(experiment)
    missing_cols = [c for c in req_cols if c not in rows[0]]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    default_metrics = EXPERIMENT_DEFAULT_METRICS.get(experiment, EXPERIMENT_DEFAULT_METRICS["overall"])
    selected_metrics = metrics_subset or default_metrics

    missing_metrics = [m for m in selected_metrics if m not in rows[0]]
    if missing_metrics:
        raise KeyError(f"Missing metrics for export: {missing_metrics}")

    if experiment in ("overall", "ablation"):
        req_paper_missing = [m for m in REQUIRED_PAPER_METRICS if m not in rows[0]]
        if req_paper_missing:
            raise KeyError(f"Missing required paper metrics: {req_paper_missing}")

    grouped: OrderedDict[str, list[dict]] = OrderedDict()
    for method in method_order:
        matched = [r for r in rows if r.get("method") == method]
        if matched:
            grouped[method] = matched
    for r in rows:
        if r.get("method") not in grouped:
            grouped.setdefault(r["method"], []).append(r)

    aggregated_rows = []
    for method, grows in grouped.items():
        base = {"method": method, "n_seeds": len(grows)}
        if experiment == "sensitivity":
            pass
        for metric in selected_metrics:
            vals = [_float(g[metric]) for g in grows]
            m = mean(vals)
            s = pstdev(vals) if len(vals) > 1 else 0.0
            base[metric] = _format(m, s)
            base[f"{metric}__mean"] = f"{m:.10f}"
            base[f"{metric}__std"] = f"{s:.10f}"
        aggregated_rows.append(base)

    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment}_{instance}"
    raw_csv = table_dir / f"{prefix}_raw.csv"
    aggregated_csv = table_dir / f"{prefix}.csv"
    tex_path = table_dir / f"{prefix}.tex"
    meta_json = table_dir / f"{prefix}.json"

    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    agg_fields = ["method", "n_seeds"] + [m for metric in selected_metrics for m in (metric, f"{metric}__mean", f"{metric}__std")]
    with aggregated_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=agg_fields)
        w.writeheader(); w.writerows(aggregated_rows)

    latex_cols = "l" + "c" * len(selected_metrics)
    header = ["Method"] + [f"{DISPLAY.get(m, (m, ''))[0]} ({DISPLAY.get(m, ('',''))[1]})".strip() for m in selected_metrics]
    lines = [f"\\begin{{tabular}}{{{latex_cols}}}", " ".join([header[0], "&", " & ".join(header[1:]), "\\\\"]), "\\hline"]
    for row in aggregated_rows:
        vals = [row[metric] for metric in selected_metrics]
        lines.append(f"{row['method']} & " + " & ".join(vals) + " \\")
    lines.append("\\end{tabular}")
    tex_path.write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "experiment": experiment,
        "instance": instance,
        "source_files": [str(summary_csv)],
        "method_order": method_order,
        "selected_metrics": selected_metrics,
        "config_snapshot": config_snapshot or {},
        "row_count": len(rows),
    }
    meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return ExportResult(str(raw_csv), str(aggregated_csv), str(tex_path), str(meta_json))
