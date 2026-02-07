from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATA_PATH = Path("avito_lenta_dataset.csv")
STEP5_PATH = Path("analysis") / "step5_outputs" / "top1000_candidates.csv"
STEP6_PATH = Path("analysis") / "step6_outputs" / "top1000_candidates_step6_text.csv"
OUT_DIR = Path("analysis") / "step7_outputs"
SUMMARY_MD = Path("analysis") / "step7_summary.md"

CASE_CATEGORIES = ["Авто", "Недвижимость", "Электроника", "Хобби и отдых"]

# Engagement proxy for pilot-readiness.
ENGAGEMENT_COLS_HIGH = [
    "questions_per_ad_avg",
    "contacts_per_ad_avg",
    "comments_per_ad_avg",
    "comment_sentiment_avg",
    "followers_count",
    "avg_views_per_day",
    "deal_success_rate",
]
ENGAGEMENT_COL_LOW = "response_time_minutes"  # lower is better


def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values=["NULL"], dtype={"user_id": "string"})
    for col in ["registration_date", "last_activity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    valid_id_mask = df["user_id"].str.fullmatch(r"[0-9a-f]{16}", na=False)
    df = df[valid_id_mask].copy()

    # Keep the most recent record if user_id duplicates appear.
    df = (
        df.sort_values(["user_id", "last_activity_date", "registration_date"], ascending=[True, False, False])
        .drop_duplicates(subset=["user_id"], keep="first")
        .copy()
    )
    return df


def strict_segment(sellers: pd.DataFrame) -> pd.DataFrame:
    return sellers[
        (sellers["total_ads_last_12m"] >= sellers["total_ads_last_12m"].quantile(0.75))
        & (sellers["avg_ad_lifetime_days"] <= sellers["avg_ad_lifetime_days"].quantile(0.25))
        & (sellers["questions_per_ad_avg"] >= sellers["questions_per_ad_avg"].quantile(0.75))
        & (sellers["rating_seller"] >= sellers["rating_seller"].quantile(0.75))
        & (sellers["reviews_count"] >= sellers["reviews_count"].quantile(0.5))
    ].copy()


def relaxed_segment(sellers: pd.DataFrame) -> pd.DataFrame:
    return sellers[
        (sellers["total_ads_last_12m"] >= sellers["total_ads_last_12m"].quantile(0.65))
        & (sellers["avg_ad_lifetime_days"] <= sellers["avg_ad_lifetime_days"].quantile(0.35))
        & (sellers["questions_per_ad_avg"] >= sellers["questions_per_ad_avg"].quantile(0.65))
        & (sellers["rating_seller"] >= 4.8)
        & (sellers["reviews_count"] >= 10)
    ].copy()


def category_metrics(seg: pd.DataFrame, sellers: pd.DataFrame, scenario: str) -> pd.DataFrame:
    base_share = sellers["ad_category_1"].value_counts(normalize=True)
    base_means = sellers[ENGAGEMENT_COLS_HIGH + [ENGAGEMENT_COL_LOW]].mean()

    agg = {"user_id": "count", **{c: "mean" for c in ENGAGEMENT_COLS_HIGH + [ENGAGEMENT_COL_LOW]}}
    table = seg.groupby("ad_category_1").agg(agg).rename(columns={"user_id": "users"}).reset_index()

    table["scenario"] = scenario
    table["seg_users"] = len(seg)
    table["seg_share"] = table["users"] / len(seg)
    table["base_seller_share"] = table["ad_category_1"].map(base_share)
    table["concentration_idx"] = table["seg_share"] / table["base_seller_share"]

    for col in ENGAGEMENT_COLS_HIGH:
        table[f"{col}_idx"] = table[col] / base_means[col]
    table["response_speed_idx"] = base_means[ENGAGEMENT_COL_LOW] / table[ENGAGEMENT_COL_LOW]

    engagement_idx_cols = [f"{c}_idx" for c in ENGAGEMENT_COLS_HIGH] + ["response_speed_idx"]
    table["engagement_idx"] = table[engagement_idx_cols].mean(axis=1)
    table["priority_score"] = table["concentration_idx"] * table["engagement_idx"]

    return table


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    users = load_clean_data(DATA_PATH)
    sellers = users[users["is_seller"] == 1].copy()

    strict = strict_segment(sellers)
    relaxed = relaxed_segment(sellers)

    step5_ids = set(pd.read_csv(STEP5_PATH, dtype={"user_id": "string"})["user_id"])
    step6_ids = set(pd.read_csv(STEP6_PATH, dtype={"user_id": "string"})["user_id"])

    step5_top1000 = sellers[sellers["user_id"].isin(step5_ids)].copy()
    step6_top1000 = sellers[sellers["user_id"].isin(step6_ids)].copy()

    segments: Dict[str, pd.DataFrame] = {
        "strict": strict,
        "relaxed": relaxed,
        "step5_top1000": step5_top1000,
        "step6_top1000": step6_top1000,
    }

    # Share table by definition.
    share_rows: List[dict] = []
    for name, seg in segments.items():
        share_rows.append(
            {
                "segment": name,
                "users": len(seg),
                "share_of_total_users": len(seg) / len(users),
                "share_of_sellers": len(seg) / len(sellers),
            }
        )
    share_df = pd.DataFrame(share_rows)
    share_df.to_csv(OUT_DIR / "pro_user_share_by_definition.csv", index=False)

    # Category metrics across scenarios.
    all_cat = pd.concat([category_metrics(seg, sellers, name) for name, seg in segments.items()], ignore_index=True)
    all_cat.to_csv(OUT_DIR / "category_metrics_all_scenarios.csv", index=False)

    case_cat = all_cat[all_cat["ad_category_1"].isin(CASE_CATEGORIES)].copy()
    case_cat.to_csv(OUT_DIR / "case_category_metrics.csv", index=False)

    # Robustness of category choice across scenarios.
    case_cat["rank_in_scenario"] = case_cat.groupby("scenario")["priority_score"].rank(ascending=False, method="min")
    robust = (
        case_cat.groupby("ad_category_1")
        .agg(
            mean_priority_score=("priority_score", "mean"),
            mean_concentration_idx=("concentration_idx", "mean"),
            mean_engagement_idx=("engagement_idx", "mean"),
            top1_count=("rank_in_scenario", lambda x: int((x == 1).sum())),
            avg_rank=("rank_in_scenario", "mean"),
        )
        .sort_values(["mean_priority_score", "top1_count"], ascending=[False, False])
        .reset_index()
    )
    robust.to_csv(OUT_DIR / "case_category_robustness.csv", index=False)

    recommended_category = robust.iloc[0]["ad_category_1"]

    # Recommendation profile based on final step6 top-1000 segment.
    rec_seg = step6_top1000[step6_top1000["ad_category_1"] == recommended_category].copy()

    profile_metrics = [
        "total_ads_last_12m",
        "total_deals_last_12m",
        "questions_per_ad_avg",
        "contacts_per_ad_avg",
        "comments_per_ad_avg",
        "comment_sentiment_avg",
        "rating_seller",
        "reviews_count",
        "deal_success_rate",
        "followers_count",
        "avg_views_per_day",
        "response_time_minutes",
    ]

    profile_rows = []
    for col in profile_metrics:
        rec_mean = rec_seg[col].mean()
        base_mean = sellers[col].mean()
        if col == "response_time_minutes":
            lift = base_mean / rec_mean
        else:
            lift = rec_mean / base_mean
        profile_rows.append(
            {
                "metric": col,
                "recommended_mean": rec_mean,
                "seller_base_mean": base_mean,
                "lift_or_speedup": lift,
            }
        )
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(OUT_DIR / "recommended_segment_profile.csv", index=False)

    marketing_mix = (
        rec_seg["user_segment_marketing"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
        .rename(columns={"index": "user_segment_marketing"})
    )
    marketing_mix.to_csv(OUT_DIR / "recommended_segment_marketing_mix.csv", index=False)

    # Final summary.
    step6_case = case_cat[case_cat["scenario"] == "step6_top1000"].copy().sort_values("priority_score", ascending=False)
    rec_row = step6_case[step6_case["ad_category_1"] == recommended_category].iloc[0]

    summary_lines = [
        "# Шаг 7: Выбор сегмента для пилота",
        "",
        "## 1) Facts from data",
        f"- Общая база после очистки: {len(users):,} пользователей; sellers: {len(sellers):,}.",
        f"- Strict сегмент: {len(strict)} ({len(strict)/len(users)*100:.3f}% от всех; {len(strict)/len(sellers)*100:.3f}% от sellers).",
        f"- Relaxed сегмент: {len(relaxed)} ({len(relaxed)/len(users)*100:.3f}% от всех; {len(relaxed)/len(sellers)*100:.3f}% от sellers).",
        f"- Финальный step6 top-1000: {len(step6_top1000)} ({len(step6_top1000)/len(users)*100:.3f}% от всех; {len(step6_top1000)/len(sellers)*100:.3f}% от sellers).",
        "",
        "## 2) Категории из условия кейса (Авто/Недвижимость/Электроника/Хобби)",
        "- Метрики для выбора: `concentration_idx` (доля в сегменте / доля у sellers), `engagement_idx` (сводный индекс вовлеченности), `priority_score = concentration_idx * engagement_idx`.",
        "",
        "### Step6 top-1000 snapshot",
    ]

    for _, row in step6_case.iterrows():
        summary_lines.append(
            f"- {row['ad_category_1']}: users={int(row['users'])}, concentration_idx={row['concentration_idx']:.3f}, engagement_idx={row['engagement_idx']:.3f}, priority_score={row['priority_score']:.3f}."
        )

    summary_lines.extend(
        [
            "",
            "## 3) Recommendation",
            f"- Рекомендованный сегмент для первой волны пилота: **Про-пользователи категории '{recommended_category}'** (на базе step6 top-1000).",
            f"- Обоснование: в step6 `{recommended_category}` имеет users={int(rec_row['users'])}, concentration_idx={rec_row['concentration_idx']:.3f}, engagement_idx={rec_row['engagement_idx']:.3f}, priority_score={rec_row['priority_score']:.3f}.",
            f"- Устойчивость: `{recommended_category}` занимает 1-е место в {int(robust.loc[robust['ad_category_1']==recommended_category, 'top1_count'].iloc[0])} из {len(segments)} сценариев, средний ранг={robust.loc[robust['ad_category_1']==recommended_category, 'avg_rank'].iloc[0]:.2f}.",
            "",
            "## 4) ASSUMPTIONS",
            "- ASSUMPTION: `priority_score = concentration_idx * engagement_idx` является рабочим прокси пилотного потенциала (сетевой эффект + шанс на вовлечение).",
            "- ASSUMPTION: step6 top-1000 — лучший operational proxy для 'первых блогеров' на текущем этапе, т.к. в нем учтены и поведенческие, и текстовые маркеры.",
            "",
            "## Артефакты",
            "- `analysis/step7_outputs/pro_user_share_by_definition.csv`",
            "- `analysis/step7_outputs/category_metrics_all_scenarios.csv`",
            "- `analysis/step7_outputs/case_category_metrics.csv`",
            "- `analysis/step7_outputs/case_category_robustness.csv`",
            "- `analysis/step7_outputs/recommended_segment_profile.csv`",
            "- `analysis/step7_outputs/recommended_segment_marketing_mix.csv`",
        ]
    )

    SUMMARY_MD.write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
