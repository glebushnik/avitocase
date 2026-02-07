from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATA_PATH = Path("avito_lenta_dataset.csv")
OUT_DIR = Path("analysis") / "step5_outputs"
SUMMARY_MD = Path("analysis") / "step5_summary.md"
TOP_K = 1000

# Metrics for transparent ProScore (higher is better unless noted)
METRICS_CONFIG = {
    "total_ads_last_12m": {"weight": 0.20, "higher_is_better": True, "label": "activity_volume"},
    "avg_ad_lifetime_days": {"weight": 0.13, "higher_is_better": False, "label": "ad_velocity"},
    "questions_per_ad_avg": {"weight": 0.17, "higher_is_better": True, "label": "engagement_depth"},
    "rating_seller": {"weight": 0.14, "higher_is_better": True, "label": "seller_trust"},
    "reviews_count": {"weight": 0.10, "higher_is_better": True, "label": "social_proof"},
    "response_time_minutes": {"weight": 0.08, "higher_is_better": False, "label": "responsiveness"},
    "total_deals_last_12m": {"weight": 0.08, "higher_is_better": True, "label": "deal_capacity"},
    "deal_success_rate": {"weight": 0.06, "higher_is_better": True, "label": "deal_quality"},
    "contacts_per_ad_avg": {"weight": 0.04, "higher_is_better": True, "label": "contact_conversion"},
}

# Base eligibility to remove very low-activity accounts from candidate ranking
ELIGIBILITY_SCENARIOS = {
    "lenient": {"ads_min": 5, "reviews_min": 3, "questions_min": 0.3},
    "base": {"ads_min": 10, "reviews_min": 5, "questions_min": 0.5},
    "strict": {"ads_min": 20, "reviews_min": 10, "questions_min": 1.0},
}

# Weight sensitivity (same eligible pool, different business priorities)
WEIGHT_SCENARIOS = {
    "balanced_base": {
        "total_ads_last_12m": 0.20,
        "avg_ad_lifetime_days": 0.13,
        "questions_per_ad_avg": 0.17,
        "rating_seller": 0.14,
        "reviews_count": 0.10,
        "response_time_minutes": 0.08,
        "total_deals_last_12m": 0.08,
        "deal_success_rate": 0.06,
        "contacts_per_ad_avg": 0.04,
    },
    "engagement_heavy": {
        "total_ads_last_12m": 0.16,
        "avg_ad_lifetime_days": 0.10,
        "questions_per_ad_avg": 0.24,
        "rating_seller": 0.10,
        "reviews_count": 0.08,
        "response_time_minutes": 0.12,
        "total_deals_last_12m": 0.06,
        "deal_success_rate": 0.04,
        "contacts_per_ad_avg": 0.10,
    },
    "trust_heavy": {
        "total_ads_last_12m": 0.15,
        "avg_ad_lifetime_days": 0.10,
        "questions_per_ad_avg": 0.12,
        "rating_seller": 0.22,
        "reviews_count": 0.16,
        "response_time_minutes": 0.08,
        "total_deals_last_12m": 0.08,
        "deal_success_rate": 0.07,
        "contacts_per_ad_avg": 0.02,
    },
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values=["NULL"], dtype={"user_id": "string"})
    for col in ["registration_date", "last_activity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_users(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    valid_id_mask = df["user_id"].str.fullmatch(r"[0-9a-f]{16}", na=False)
    invalid_rows = int((~valid_id_mask).sum())

    cleaned = df[valid_id_mask].copy()
    before_dedup = len(cleaned)

    cleaned = (
        cleaned.sort_values(["user_id", "last_activity_date", "registration_date"], ascending=[True, False, False])
        .drop_duplicates(subset=["user_id"], keep="first")
        .copy()
    )

    stats = {
        "rows_in": len(df),
        "rows_after_valid_id": before_dedup,
        "rows_after_dedup": len(cleaned),
        "invalid_id_rows_removed": invalid_rows,
        "duplicate_rows_removed": before_dedup - len(cleaned),
    }
    return cleaned, stats


def percentile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    # Rank-based score in [0,1]. Missing values are set to neutral 0.5.
    ranked = series.rank(method="average", pct=True, ascending=higher_is_better)
    return ranked.fillna(0.5)


def add_metric_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    scored = df.copy()
    for metric, conf in METRICS_CONFIG.items():
        col = f"score_{metric}"
        scored[col] = percentile_score(scored[metric], higher_is_better=conf["higher_is_better"])

    scored["pro_score_raw"] = 0.0
    for metric, weight in weights.items():
        scored["pro_score_raw"] += scored[f"score_{metric}"] * weight

    scored["pro_score"] = (scored["pro_score_raw"] * 100).round(2)
    scored = scored.sort_values("pro_score", ascending=False).copy()
    scored["rank"] = np.arange(1, len(scored) + 1)
    return scored


def apply_eligibility(df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    mask = (
        (df["total_ads_last_12m"].fillna(0) >= scenario["ads_min"])
        & (df["reviews_count"].fillna(0) >= scenario["reviews_min"])
        & (df["questions_per_ad_avg"].fillna(0) >= scenario["questions_min"])
    )
    return df[mask].copy()


def top_ids(df: pd.DataFrame, k: int = TOP_K) -> pd.Index:
    return df.head(min(k, len(df)))["user_id"]


def overlap_metrics(base_ids: pd.Index, other_ids: pd.Index) -> dict:
    base_set = set(base_ids)
    other_set = set(other_ids)
    inter = len(base_set.intersection(other_set))
    union = len(base_set.union(other_set))
    return {
        "overlap_count": inter,
        "overlap_share_of_base": round(inter / len(base_set), 4) if base_set else np.nan,
        "jaccard": round(inter / union, 4) if union else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(DATA_PATH)
    df, clean_stats = clean_users(df_raw)

    sellers = df[df["is_seller"] == 1].copy()

    # Base scoring pipeline (balanced weights + base eligibility)
    base_eligible = apply_eligibility(sellers, ELIGIBILITY_SCENARIOS["base"])
    scored_base = add_metric_scores(base_eligible, WEIGHT_SCENARIOS["balanced_base"])

    top1000 = scored_base.head(min(TOP_K, len(scored_base))).copy()

    # Persist top candidates with interpretable component scores
    keep_cols = [
        "rank",
        "user_id",
        "pro_score",
        "pro_score_raw",
        "user_segment_marketing",
        "user_city",
        "user_region",
        "ad_category_1",
        "ad_category_2",
        "ad_category_3",
        "total_ads_last_12m",
        "avg_ad_lifetime_days",
        "questions_per_ad_avg",
        "rating_seller",
        "reviews_count",
        "response_time_minutes",
        "total_deals_last_12m",
        "deal_success_rate",
        "contacts_per_ad_avg",
        "followers_count",
        "avg_views_per_day",
    ] + [f"score_{m}" for m in METRICS_CONFIG.keys()]

    top1000[keep_cols].to_csv(OUT_DIR / "top1000_candidates.csv", index=False)

    # Threshold sensitivity (same weights, varying eligibility gates)
    base_top_ids = top_ids(scored_base)
    threshold_rows = []
    for scenario_name, gates in ELIGIBILITY_SCENARIOS.items():
        eligible = apply_eligibility(sellers, gates)
        scored = add_metric_scores(eligible, WEIGHT_SCENARIOS["balanced_base"])
        ids = top_ids(scored)
        row = {
            "scenario": scenario_name,
            "eligible_users": len(eligible),
            "top_k_used": min(TOP_K, len(scored)),
            "ads_min": gates["ads_min"],
            "reviews_min": gates["reviews_min"],
            "questions_min": gates["questions_min"],
        }
        row.update(overlap_metrics(base_top_ids, ids))
        threshold_rows.append(row)

    threshold_sensitivity = pd.DataFrame(threshold_rows)
    threshold_sensitivity.to_csv(OUT_DIR / "threshold_sensitivity.csv", index=False)

    # Weight sensitivity (same eligibility, varying weights)
    weight_rows = []
    for scenario_name, weights in WEIGHT_SCENARIOS.items():
        scored = add_metric_scores(base_eligible, weights)
        ids = top_ids(scored)
        row = {
            "scenario": scenario_name,
            "eligible_users": len(base_eligible),
            "top_k_used": min(TOP_K, len(scored)),
            "weights_sum": round(sum(weights.values()), 6),
        }
        row.update(overlap_metrics(base_top_ids, ids))
        weight_rows.append(row)

    weight_sensitivity = pd.DataFrame(weight_rows)
    weight_sensitivity.to_csv(OUT_DIR / "weight_sensitivity.csv", index=False)

    # Distribution diagnostics for top1000
    category_dist = (
        top1000["ad_category_1"]
        .value_counts(dropna=False)
        .rename("top1000_count")
        .reset_index()
        .rename(columns={"index": "ad_category_1"})
    )
    category_dist["top1000_share"] = (category_dist["top1000_count"] / len(top1000)).round(4)

    base_dist = sellers["ad_category_1"].value_counts(normalize=True, dropna=False)
    category_dist["seller_base_share"] = category_dist["ad_category_1"].map(base_dist).fillna(0).round(4)
    category_dist["overindex_vs_sellers"] = (
        (category_dist["top1000_share"] / category_dist["seller_base_share"].replace(0, np.nan)).round(2)
    )
    category_dist.to_csv(OUT_DIR / "top1000_category_distribution.csv", index=False)

    segment_dist = (
        top1000["user_segment_marketing"]
        .value_counts(dropna=False)
        .rename("top1000_count")
        .reset_index()
        .rename(columns={"index": "user_segment_marketing"})
    )
    segment_dist["top1000_share"] = (segment_dist["top1000_count"] / len(top1000)).round(4)

    sellers_segment_share = sellers["user_segment_marketing"].value_counts(normalize=True, dropna=False)
    segment_dist["seller_base_share"] = segment_dist["user_segment_marketing"].map(sellers_segment_share).fillna(0).round(4)
    segment_dist["overindex_vs_sellers"] = (
        (segment_dist["top1000_share"] / segment_dist["seller_base_share"].replace(0, np.nan)).round(2)
    )
    segment_dist.to_csv(OUT_DIR / "top1000_segment_distribution.csv", index=False)

    # Human-readable summary
    top1 = top1000.iloc[0]
    top10_score_min = float(top1000.head(10)["pro_score"].min())
    top1000_score_min = float(top1000["pro_score"].min())
    top1000_score_med = float(top1000["pro_score"].median())

    summary_lines = [
        "# Шаг 5: Scoring + Ranking Top-1000 кандидатов",
        "",
        "## 1) Facts from data",
        f"- После очистки ID и дедупликации: {clean_stats['rows_after_dedup']:,} строк (удалено невалидных ID: {clean_stats['invalid_id_rows_removed']}, удалено дублей ID: {clean_stats['duplicate_rows_removed']}).",
        f"- Sellers после очистки: {len(sellers):,}.",
        f"- Eligible pool (base thresholds): {len(base_eligible):,}.",
        f"- Сформирован рейтинг top-{min(TOP_K, len(scored_base))} кандидатов: `analysis/step5_outputs/top1000_candidates.csv`.",
        "",
        "## 2) Scoring rule (transparent)",
        "- `ProScore` = взвешенная сумма перцентильных component-скоров (0..1), далее масштаб в 0..100.",
        "- Перцентильные скоры считаются только по sellers; для missing у компонента используется нейтральное значение 0.5.",
        "",
        "### Метрики и веса",
    ]

    for metric, conf in METRICS_CONFIG.items():
        direction = "higher_is_better" if conf["higher_is_better"] else "lower_is_better"
        summary_lines.append(f"- `{metric}`: weight={conf['weight']:.2f}, {direction}")

    summary_lines.extend(
        [
            "",
            "## 3) Base eligibility rule",
            f"- `is_seller=1` AND `total_ads_last_12m >= {ELIGIBILITY_SCENARIOS['base']['ads_min']}` AND `reviews_count >= {ELIGIBILITY_SCENARIOS['base']['reviews_min']}` AND `questions_per_ad_avg >= {ELIGIBILITY_SCENARIOS['base']['questions_min']}`.",
            "",
            "## 4) Ranking snapshot",
            f"- Top-1 ProScore: {top1['pro_score']:.2f} (user_id={top1['user_id']}).",
            f"- Минимальный ProScore в top-10: {top10_score_min:.2f}.",
            f"- Медиана ProScore в top-1000: {top1000_score_med:.2f}.",
            f"- Минимальный ProScore в top-1000: {top1000_score_min:.2f}.",
            "",
            "## 5) Sensitivity checks",
            "### Threshold sensitivity (same weights)",
        ]
    )

    for _, row in threshold_sensitivity.iterrows():
        summary_lines.append(
            f"- {row['scenario']}: eligible={int(row['eligible_users'])}, overlap_with_base={row['overlap_count']} ({row['overlap_share_of_base']*100:.2f}% от base top-1000), jaccard={row['jaccard']:.4f}."
        )

    summary_lines.append("")
    summary_lines.append("### Weight sensitivity (same eligibility)")
    for _, row in weight_sensitivity.iterrows():
        summary_lines.append(
            f"- {row['scenario']}: overlap_with_base={row['overlap_count']} ({row['overlap_share_of_base']*100:.2f}% от base top-1000), jaccard={row['jaccard']:.4f}."
        )

    summary_lines.extend(
        [
            "",
            "## 6) Interpretations (not facts)",
            "- Если overlap высок даже при изменении весов/порогов, ядро кандидатов устойчиво и подходит для пилотного набора.",
            "- Если overlap заметно падает при strict thresholds, значит долгосрочно понадобится расширенная воронка (вторая волна кандидатов).",
            "",
            "## 7) ASSUMPTIONS",
            "- ASSUMPTION: neutral fill (0.5) для missing component-скора приемлем, т.к. внутри sellers доля пропусков по ключевым метрикам относительно низкая.",
            "- ASSUMPTION: выбранные веса отражают продуктовый приоритет пилота (активность + вовлечение + доверие), а не долгосрочную монетизацию.",
            "",
            "## Артефакты",
            "- `analysis/step5_outputs/top1000_candidates.csv`",
            "- `analysis/step5_outputs/threshold_sensitivity.csv`",
            "- `analysis/step5_outputs/weight_sensitivity.csv`",
            "- `analysis/step5_outputs/top1000_category_distribution.csv`",
            "- `analysis/step5_outputs/top1000_segment_distribution.csv`",
        ]
    )

    SUMMARY_MD.write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
