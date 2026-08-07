# build_size_tables.py
import pandas as pd
import re
import os
import json
from io import StringIO
from pathlib import Path

# ##########################################################
# =*= COLLECT ALL CUT-TIME / ACCEPTANCE RESULTS BY SIZE =*=
# ##########################################################
__author__  = "Hedi Boukamcha"
__version__ = "1.0.0"
__license__ = "MIT"

REPO_ROOT     = Path(__file__).resolve().parent
ANALYSIS_ROOT = REPO_ROOT / "analysis"
OUTPUT_DIR    = ANALYSIS_ROOT / "results_by_size"
CSV_DIR       = OUTPUT_DIR / "csv"
UB_ROOT       = REPO_ROOT / "data" / "controlled_orders_ub" / "test"

CUT_TIME_ORDER = ["early", "middle", "late"]
INSTANCE_DIR_RE = re.compile(r"^instance_(\d+)_(early|middle|late)$")
DELTA_DIR_RE = re.compile(r"^delta_(\d+)_(\d+)$")


def parse_summary_path(summary_csv: Path) -> dict | None:
    instance_dir = summary_csv.parent.name
    variant_dir = summary_csv.parent.parent.name
    delta_dir = summary_csv.parent.parent.parent.name
    size_dir = summary_csv.parent.parent.parent.parent.name

    inst_match = INSTANCE_DIR_RE.match(instance_dir)
    delta_match = DELTA_DIR_RE.match(delta_dir)
    if not inst_match or not delta_match:
        return None

    instance_num = int(inst_match.group(1))
    cut_time_pos = inst_match.group(2)
    delta_ratio = float(f"{delta_match.group(1)}.{delta_match.group(2)}")

    return {
        "size": size_dir,
        "delta_ratio": delta_ratio,
        "cost_variant": variant_dir,
        "instance": instance_num,
        "cut_time_pos": cut_time_pos,
    }


def read_order2_csv(order2_csv: Path) -> pd.DataFrame:
    """order_2_acceptance_analysis.csv is written with one extra trailing comma
    per data row, which makes pandas shift every column by one (Pool ends up
    as the index). Strip that single spurious trailing empty field first."""
    lines = order2_csv.read_text().splitlines()
    header = lines[0]
    n_cols = len(header.split(","))
    fixed = [header]
    for line in lines[1:]:
        if not line:
            continue
        fields = line.split(",")
        if len(fields) == n_cols + 1 and fields[-1] == "":
            line = line[:-1]
        fixed.append(line)
    return pd.read_csv(StringIO("\n".join(fixed)))


def existing_job_costs(size: str, cost_variant: str, instance: int, cut_time_pos: str) -> dict[str, float]:
    """Cost of each existing job (order 1), keyed as 'J1', 'J2', ... to match
    the 'job' column of order_2_acceptance_analysis.csv."""
    json_path = UB_ROOT / size / cost_variant / f"instance_{instance}_{cut_time_pos}.json"
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        data = json.load(f)
    existing_jobs = data["orders"][0]["jobs"]
    return {f"J{i + 1}": j["cost"] for i, j in enumerate(existing_jobs)}


def weighted_sum_delta_tjE(order2_csv: Path, job_costs: dict[str, float]) -> float | None:
    """Cost-weighted sum of Diff_Tj over existing jobs (Pool == 'Existants'):
    sum(cost_j * (Tj_final - Tj_initial)) = total weighted tardiness shift
    caused by accepting new orders."""
    if not order2_csv.exists() or not job_costs:
        return None
    df = read_order2_csv(order2_csv)
    existing = df.loc[df["Pool"] == "Existants", ["job", "Diff_Tj"]].copy()
    existing["Diff_Tj"] = pd.to_numeric(existing["Diff_Tj"], errors="coerce")
    existing = existing.dropna(subset=["Diff_Tj"])
    if existing.empty:
        return None
    weighted = existing["Diff_Tj"] * existing["job"].map(job_costs)
    return weighted.sum() if not weighted.isna().all() else None


def build_master_table() -> pd.DataFrame:
    rows = []
    for summary_csv in ANALYSIS_ROOT.glob("*/delta_*/*/instance_*/summary.csv"):
        parsed = parse_summary_path(summary_csv)
        if parsed is None:
            print(f"[WARN] skipped (unexpected path shape): {summary_csv}")
            continue
        df = pd.read_csv(summary_csv)
        if df.empty:
            print(f"[WARN] empty summary.csv: {summary_csv}")
            continue
        row = df.iloc[0].to_dict()
        row.update(parsed)

        order2_csv = summary_csv.parent / "order_2_acceptance_analysis.csv"
        job_costs = existing_job_costs(parsed["size"], parsed["cost_variant"], parsed["instance"], parsed["cut_time_pos"])
        row["delta_tjE"] = weighted_sum_delta_tjE(order2_csv, job_costs)

        rows.append(row)

    if not rows:
        raise RuntimeError(f"No summary.csv files found under {ANALYSIS_ROOT}")

    master = pd.DataFrame(rows)
    master["cut_time_pos"] = pd.Categorical(master["cut_time_pos"], categories=CUT_TIME_ORDER, ordered=True)

    front_cols = ["size", "cost_variant", "delta_ratio", "instance", "cut_time_pos"]
    metric_cols = [c for c in master.columns if c not in front_cols and c not in ("scenario", "inst", "variant")]
    master = master[front_cols + metric_cols]
    master = master.sort_values(["size", "cost_variant", "delta_ratio", "instance", "cut_time_pos"])
    return master.reset_index(drop=True)


def write_tables_per_size_and_scenario(master: pd.DataFrame) -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for (size, cost_variant), df_group in master.groupby(["size", "cost_variant"], observed=True):
        out_csv = CSV_DIR / f"results_{size}_{cost_variant}.csv"
        df_group.to_csv(out_csv, index=False)
        print(f"[INFO] {out_csv} -> {len(df_group)} rows")

# RUN: python3 build_analysis_tables.py
if __name__ == "__main__":
    master_df = build_master_table()
    write_tables_per_size_and_scenario(master_df)
    master_df.to_csv(CSV_DIR / "results_all_sizes.csv", index=False)
    print(f"[INFO] {CSV_DIR / 'results_all_sizes.csv'} -> {len(master_df)} rows (all sizes combined)")
