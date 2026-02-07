from __future__ import annotations

from pathlib import Path
import pandas as pd


DATA_PATH = Path("avito_lenta_dataset.csv")
OUT_DIR = Path("analysis") / "step2_4_outputs"
SUMMARY_MD = Path("analysis") / "step2_4_summary.md"

KEY_QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99]
CATEGORY_FOCUS = ["Авто", "Недвижимость", "Электроника", "Хобби и отдых"]


def pct(value: float) -> float:
    return round(value * 100, 2)


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values=["NULL"], dtype={"user_id": "string"})
    for col in ["registration_date", "last_activity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def build_strict_rule(sellers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {
        "total_ads_last_12m_p75": float(sellers["total_ads_last_12m"].quantile(0.75)),
        "avg_ad_lifetime_days_p25": float(sellers["avg_ad_lifetime_days"].quantile(0.25)),
        "questions_per_ad_avg_p75": float(sellers["questions_per_ad_avg"].quantile(0.75)),
        "rating_seller_p75": float(sellers["rating_seller"].quantile(0.75)),
        "reviews_count_p50": float(sellers["reviews_count"].quantile(0.5)),
    }

    pro = sellers[
        (sellers["total_ads_last_12m"] >= thresholds["total_ads_last_12m_p75"])
        & (sellers["avg_ad_lifetime_days"] <= thresholds["avg_ad_lifetime_days_p25"])
        & (sellers["questions_per_ad_avg"] >= thresholds["questions_per_ad_avg_p75"])
        & (sellers["rating_seller"] >= thresholds["rating_seller_p75"])
        & (sellers["reviews_count"] >= thresholds["reviews_count_p50"])
    ].copy()

    return pro, thresholds


def build_relaxed_rule(sellers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    thresholds = {
        "total_ads_last_12m_p65": float(sellers["total_ads_last_12m"].quantile(0.65)),
        "avg_ad_lifetime_days_p35": float(sellers["avg_ad_lifetime_days"].quantile(0.35)),
        "questions_per_ad_avg_p65": float(sellers["questions_per_ad_avg"].quantile(0.65)),
        "rating_seller_fixed": 4.8,
        "reviews_count_fixed": 10.0,
    }

    pro = sellers[
        (sellers["total_ads_last_12m"] >= thresholds["total_ads_last_12m_p65"])
        & (sellers["avg_ad_lifetime_days"] <= thresholds["avg_ad_lifetime_days_p35"])
        & (sellers["questions_per_ad_avg"] >= thresholds["questions_per_ad_avg_p65"])
        & (sellers["rating_seller"] >= thresholds["rating_seller_fixed"])
        & (sellers["reviews_count"] >= thresholds["reviews_count_fixed"])
    ].copy()

    return pro, thresholds


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA_PATH)

    valid_id_mask = df["user_id"].str.fullmatch(r"[0-9a-f]{16}", na=False)
    sellers = df[df["is_seller"] == 1].copy()

    # Step 2: data audit tables
    missing = (
        (df.isna().mean() * 100)
        .sort_values(ascending=False)
        .rename("missing_pct")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    missing.to_csv(OUT_DIR / "missingness.csv", index=False)

    logic_checks = pd.DataFrame(
        {
            "check": [
                "active_ads_current > total_ads_last_12m",
                "deal_success_rate outside [0,1]",
                "positive_reviews_percent outside [0,100]",
                "last_activity_date < registration_date",
                "seller_with_zero_ads",
                "non_seller_with_ads",
                "users_with_deals_but_not_buyer",
            ],
            "count": [
                int((df["active_ads_current"] > df["total_ads_last_12m"]).sum()),
                int(((df["deal_success_rate"] < 0) | (df["deal_success_rate"] > 1)).sum()),
                int(
                    (
                        (df["positive_reviews_percent"].dropna() < 0)
                        | (df["positive_reviews_percent"].dropna() > 100)
                    ).sum()
                ),
                int((df["last_activity_date"] < df["registration_date"]).sum()),
                int(((df["is_seller"] == 1) & (df["total_ads_last_12m"] == 0)).sum()),
                int(((df["is_seller"] == 0) & (df["total_ads_last_12m"] > 0)).sum()),
                int(((df["is_buyer"] == 0) & (df["total_deals_last_12m"] > 0)).sum()),
            ],
        }
    )
    logic_checks.to_csv(OUT_DIR / "logic_checks.csv", index=False)

    # Step 3: EDA tables
    quantile_cols = [
        "total_ads_last_12m",
        "active_ads_current",
        "avg_ad_lifetime_days",
        "total_deals_last_12m",
        "deal_success_rate",
        "avg_deal_amount_rub",
        "total_revenue_last_12m",
        "rating_seller",
        "reviews_count",
        "response_time_minutes",
        "views_per_ad_avg",
        "contacts_per_ad_avg",
        "questions_per_ad_avg",
        "followers_count",
        "avg_views_per_day",
    ]
    quantiles = (
        df[quantile_cols]
        .quantile(KEY_QUANTILES)
        .T.reset_index()
        .rename(columns={"index": "metric"})
        .rename(columns={0.5: "p50", 0.75: "p75", 0.9: "p90", 0.95: "p95", 0.99: "p99"})
    )
    quantiles["max"] = [df[col].max() for col in quantile_cols]
    quantiles.to_csv(OUT_DIR / "quantiles.csv", index=False)

    segment_metrics = (
        df.groupby("user_segment_marketing")
        .agg(
            users=("user_id", "count"),
            seller_share=("is_seller", "mean"),
            buyer_share=("is_buyer", "mean"),
            ads_mean=("total_ads_last_12m", "mean"),
            active_ads_mean=("active_ads_current", "mean"),
            deals_mean=("total_deals_last_12m", "mean"),
            questions_mean=("questions_per_ad_avg", "mean"),
            rating_mean=("rating_seller", "mean"),
            revenue_mean=("total_revenue_last_12m", "mean"),
            followers_mean=("followers_count", "mean"),
        )
        .sort_values("users", ascending=False)
        .reset_index()
    )
    segment_metrics.to_csv(OUT_DIR / "segment_metrics.csv", index=False)

    category_focus_metrics = (
        df[df["ad_category_1"].isin(CATEGORY_FOCUS)]
        .groupby("ad_category_1")
        .agg(
            users=("user_id", "count"),
            seller_share=("is_seller", "mean"),
            ads_mean=("total_ads_last_12m", "mean"),
            deals_mean=("total_deals_last_12m", "mean"),
            questions_mean=("questions_per_ad_avg", "mean"),
            rating_mean=("rating_seller", "mean"),
            followers_mean=("followers_count", "mean"),
            views_per_day_mean=("avg_views_per_day", "mean"),
        )
        .sort_values("users", ascending=False)
        .reset_index()
    )
    category_focus_metrics.to_csv(OUT_DIR / "category_focus_metrics.csv", index=False)

    # Step 4: baseline pro-user rules
    strict_pro, strict_thr = build_strict_rule(sellers)
    relaxed_pro, relaxed_thr = build_relaxed_rule(sellers)

    rule_summary = pd.DataFrame(
        [
            {
                "rule": "strict",
                "count": len(strict_pro),
                "share_total_pct": pct(len(strict_pro) / len(df)),
                "share_sellers_pct": pct(len(strict_pro) / len(sellers)),
                **strict_thr,
            },
            {
                "rule": "relaxed",
                "count": len(relaxed_pro),
                "share_total_pct": pct(len(relaxed_pro) / len(df)),
                "share_sellers_pct": pct(len(relaxed_pro) / len(sellers)),
                **relaxed_thr,
            },
        ]
    )
    rule_summary.to_csv(OUT_DIR / "pro_user_rule_summary.csv", index=False)

    strict_overindex = (
        strict_pro["ad_category_1"].value_counts(normalize=True)
        / df["ad_category_1"].value_counts(normalize=True)
    ).sort_values(ascending=False)
    relaxed_overindex = (
        relaxed_pro["ad_category_1"].value_counts(normalize=True)
        / df["ad_category_1"].value_counts(normalize=True)
    ).sort_values(ascending=False)

    overindex_df = pd.concat(
        [
            strict_overindex.rename("strict_overindex"),
            relaxed_overindex.rename("relaxed_overindex"),
        ],
        axis=1,
    ).reset_index().rename(columns={"index": "ad_category_1"})
    overindex_df.to_csv(OUT_DIR / "pro_user_overindex.csv", index=False)

    # Summary markdown for quick review
    role_combo = {
        "seller_share": pct((df["is_seller"] == 1).mean()),
        "buyer_share": pct((df["is_buyer"] == 1).mean()),
        "both_share": pct(((df["is_seller"] == 1) & (df["is_buyer"] == 1)).mean()),
        "seller_only_share": pct(((df["is_seller"] == 1) & (df["is_buyer"] == 0)).mean()),
        "buyer_only_share": pct(((df["is_seller"] == 0) & (df["is_buyer"] == 1)).mean()),
        "neither_share": pct(((df["is_seller"] == 0) & (df["is_buyer"] == 0)).mean()),
    }

    top_missing = missing.head(6)
    top_categories = df["ad_category_1"].value_counts().head(8)

    summary_lines = [
        "# Шаги 2-4: Data Audit + EDA + Baseline Pro-user",
        "",
        "## 1) Facts from data (Step 2: Data understanding & quality)",
        f"- Размер: **{len(df):,}** строк, **{len(df.columns)}** колонок, уникальных `user_id`: **{df['user_id'].nunique():,}**.",
        f"- Период регистраций: **{df['registration_date'].min().date()} — {df['registration_date'].max().date()}**.",
        f"- Период последней активности: **{df['last_activity_date'].min().date()} — {df['last_activity_date'].max().date()}**.",
        f"- Валидация `user_id`: валидных hex-16 id **{int(valid_id_mask.sum()):,}**, невалидных id **{int((~valid_id_mask).sum())}** (уникальных невалидных: **{df.loc[~valid_id_mask, 'user_id'].nunique()}**).",
        f"- Дубликаты: по полным строкам **{int(df.duplicated().sum())}**, по `user_id` **{int(df['user_id'].duplicated().sum())}**.",
        "",
        "### Пропуски (топ)",
    ]
    for _, row in top_missing.iterrows():
        summary_lines.append(f"- `{row['column']}`: **{row['missing_pct']:.2f}%**")

    summary_lines.extend(
        [
            "",
            "### Логические проверки",
            f"- `active_ads_current > total_ads_last_12m`: **{int(logic_checks.loc[logic_checks['check']=='active_ads_current > total_ads_last_12m','count'].iloc[0])}**.",
            f"- `deal_success_rate` вне [0,1]: **{int(logic_checks.loc[logic_checks['check']=='deal_success_rate outside [0,1]','count'].iloc[0])}**.",
            f"- `positive_reviews_percent` вне [0,100]: **{int(logic_checks.loc[logic_checks['check']=='positive_reviews_percent outside [0,100]','count'].iloc[0])}**.",
            f"- `last_activity_date < registration_date`: **{int(logic_checks.loc[logic_checks['check']=='last_activity_date < registration_date','count'].iloc[0])}**.",
            f"- `is_seller=0` и `total_ads_last_12m>0`: **{int(logic_checks.loc[logic_checks['check']=='non_seller_with_ads','count'].iloc[0])}**.",
            f"- `is_buyer=0` и `total_deals_last_12m>0`: **{int(logic_checks.loc[logic_checks['check']=='users_with_deals_but_not_buyer','count'].iloc[0])}**.",
            "",
            "## 2) EDA (Step 3)",
            f"- Ролевой микс: sellers **{role_combo['seller_share']:.2f}%**, buyers **{role_combo['buyer_share']:.2f}%**, обе роли **{role_combo['both_share']:.2f}%**.",
            f"- Только sellers: **{role_combo['seller_only_share']:.2f}%**, только buyers: **{role_combo['buyer_only_share']:.2f}%**, ни одна роль: **{role_combo['neither_share']:.2f}%**.",
            "",
            "### Основные хвосты распределений",
        ]
    )

    p = quantiles.set_index("metric")
    summary_lines.extend(
        [
            f"- `total_ads_last_12m`: p50={p.loc['total_ads_last_12m','p50']:.2f}, p90={p.loc['total_ads_last_12m','p90']:.2f}, p99={p.loc['total_ads_last_12m','p99']:.2f}, max={p.loc['total_ads_last_12m','max']:.2f}.",
            f"- `total_revenue_last_12m`: p50={p.loc['total_revenue_last_12m','p50']:.2f}, p90={p.loc['total_revenue_last_12m','p90']:.2f}, p99={p.loc['total_revenue_last_12m','p99']:.2f}, max={p.loc['total_revenue_last_12m','max']:.2f}.",
            f"- `questions_per_ad_avg`: p50={p.loc['questions_per_ad_avg','p50']:.2f}, p90={p.loc['questions_per_ad_avg','p90']:.2f}, p99={p.loc['questions_per_ad_avg','p99']:.2f}, max={p.loc['questions_per_ad_avg','max']:.2f}.",
            f"- `followers_count`: p50={p.loc['followers_count','p50']:.2f}, p90={p.loc['followers_count','p90']:.2f}, p99={p.loc['followers_count','p99']:.2f}, max={p.loc['followers_count','max']:.2f}.",
            "",
            "### Крупнейшие категории (ad_category_1)",
        ]
    )
    for cat, cnt in top_categories.items():
        summary_lines.append(f"- {cat}: **{cnt:,}**")

    summary_lines.extend(
        [
            "",
            "## 3) Baseline сегментации 'Про-пользователь' (Step 4)",
            "### Strict baseline rule",
            f"- `is_seller=1` AND `total_ads_last_12m >= {strict_thr['total_ads_last_12m_p75']:.1f}` AND `avg_ad_lifetime_days <= {strict_thr['avg_ad_lifetime_days_p25']:.1f}` AND `questions_per_ad_avg >= {strict_thr['questions_per_ad_avg_p75']:.1f}` AND `rating_seller >= {strict_thr['rating_seller_p75']:.1f}` AND `reviews_count >= {strict_thr['reviews_count_p50']:.1f}`.",
            f"- Результат strict: **{len(strict_pro)}** пользователей (**{pct(len(strict_pro)/len(df)):.2f}%** от всей базы, **{pct(len(strict_pro)/len(sellers)):.2f}%** от sellers).",
            "",
            "### Relaxed baseline rule (для целевого пула ~1000 на следующем шаге)",
            f"- `is_seller=1` AND `total_ads_last_12m >= {relaxed_thr['total_ads_last_12m_p65']:.1f}` AND `avg_ad_lifetime_days <= {relaxed_thr['avg_ad_lifetime_days_p35']:.1f}` AND `questions_per_ad_avg >= {relaxed_thr['questions_per_ad_avg_p65']:.1f}` AND `rating_seller >= {relaxed_thr['rating_seller_fixed']:.1f}` AND `reviews_count >= {relaxed_thr['reviews_count_fixed']:.0f}`.",
            f"- Результат relaxed: **{len(relaxed_pro)}** пользователей (**{pct(len(relaxed_pro)/len(df)):.2f}%** от всей базы, **{pct(len(relaxed_pro)/len(sellers)):.2f}%** от sellers).",
            "",
            "## 4) Interpretations (not facts)",
            "- Большая доля пропусков в `rating_seller`, `positive_reviews_percent`, `response_time_minutes` в основном структурная: метрики не применяются к части non-sellers.",
            "- Маркетинговые сегменты (`Новичок`, `Активный`, `Премиум` и т.д.) в текущем срезе слабо разделяют пользователей по core-операционным метрикам, поэтому для отбора кандидатов лучше опираться на поведенческие признаки.",
            "- Распределения выручки/объемов сделок сильно скошены вправо; медианные и перцентильные пороги надежнее средних для правил сегментации.",
            "",
            "## 5) ASSUMPTIONS",
            "- ASSUMPTION: `is_seller`/`is_buyer` отражают текущую роль пользователя, а исторические активности (`total_ads_last_12m`, `total_deals_last_12m`) могут включать прошлые роли; поэтому комбинации вида `is_seller=0` + `total_ads_last_12m>0` считаем не ошибкой по умолчанию.",
            "- ASSUMPTION: для baseline сегментации скорость обновления объявлений приближенно измеряется как низкий `avg_ad_lifetime_days`.",
            "",
            "## Артефакты",
            "- `analysis/step2_4_outputs/missingness.csv`",
            "- `analysis/step2_4_outputs/logic_checks.csv`",
            "- `analysis/step2_4_outputs/quantiles.csv`",
            "- `analysis/step2_4_outputs/segment_metrics.csv`",
            "- `analysis/step2_4_outputs/category_focus_metrics.csv`",
            "- `analysis/step2_4_outputs/pro_user_rule_summary.csv`",
            "- `analysis/step2_4_outputs/pro_user_overindex.csv`",
        ]
    )

    SUMMARY_MD.write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
