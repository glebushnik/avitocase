from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd
import re


DATA_PATH = Path("avito_lenta_dataset.csv")
STEP5_TOP1000_PATH = Path("analysis") / "step5_outputs" / "top1000_candidates.csv"
OUT_DIR = Path("analysis") / "step6_outputs"
SUMMARY_MD = Path("analysis") / "step6_summary.md"
TOP_K = 1000

# Reuse step5 behavior setup for consistency
BEHAVIOR_WEIGHTS = {
    "total_ads_last_12m": 0.20,
    "avg_ad_lifetime_days": 0.13,
    "questions_per_ad_avg": 0.17,
    "rating_seller": 0.14,
    "reviews_count": 0.10,
    "response_time_minutes": 0.08,
    "total_deals_last_12m": 0.08,
    "deal_success_rate": 0.06,
    "contacts_per_ad_avg": 0.04,
}

BEHAVIOR_DIRECTION = {
    "total_ads_last_12m": True,
    "avg_ad_lifetime_days": False,
    "questions_per_ad_avg": True,
    "rating_seller": True,
    "reviews_count": True,
    "response_time_minutes": False,
    "total_deals_last_12m": True,
    "deal_success_rate": True,
    "contacts_per_ad_avg": True,
}

BASE_ELIGIBILITY = {"ads_min": 10, "reviews_min": 5, "questions_min": 0.5}

# Interpretable dictionaries for stylistic markers
PROFESSIONAL_TERMS = [
    "оригинал",
    "проверено",
    "чек",
    "гарантия",
    "гарантией",
    "диагностики",
    "состояние",
    "магазина",
    "идеальное",
    "минимальными",
    "следами",
    "хорошее",
    "новое",
]

LOGISTICS_TERMS = [
    "доставка",
    "самовывоз",
    "без торга",
    "торг уместен",
    "подробности",
    "запросу",
]

CHARISMA_TERMS = [
    "пишите",
    "чат",
    "отвечу",
    "быстро",
    "срочно",
    "торг",
    "уместен",
]

EXPERTISE_TERMS = [
    "apple",
    "sony",
    "bmw",
    "kia",
    "toyota",
    "hyundai",
    "huawei",
    "adidas",
    "anex",
    "bosch",
    "lego",
    "игровые",
    "приставки",
    "запчасть",
    "шины",
    "диски",
    "программирование",
    "вождение",
    "аренда",
    "переезды",
    "музыкальные",
    "кофемашина",
    "курьер",
    "грузчики",
    "коллекционирование",
    "ноутбук",
    "смартфон",
    "квартир",
    "ремонт",
]

STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "для",
    "у",
    "к",
    "от",
    "до",
    "за",
    "из",
    "не",
    "но",
    "а",
    "то",
    "же",
    "или",
    "это",
    "как",
    "что",
    "при",
    "без",
    "под",
    "над",
    "если",
    "либо",
    "очень",
    "после",
}


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", na_values=["NULL"], dtype={"user_id": "string"})
    for col in ["registration_date", "last_activity_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_users(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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


def apply_eligibility(df: pd.DataFrame, gates: dict) -> pd.DataFrame:
    mask = (
        (df["total_ads_last_12m"].fillna(0) >= gates["ads_min"])
        & (df["reviews_count"].fillna(0) >= gates["reviews_min"])
        & (df["questions_per_ad_avg"].fillna(0) >= gates["questions_min"])
    )
    return df[mask].copy()


def pct_rank(series: pd.Series, higher_is_better: bool) -> pd.Series:
    ranked = series.rank(method="average", pct=True, ascending=higher_is_better)
    return ranked.fillna(0.5)


def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace("ё", "е", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def count_terms(text: pd.Series, terms: Iterable[str]) -> pd.Series:
    pattern = "(?:" + "|".join(re.escape(t) for t in terms) + ")"
    return text.str.count(pattern)


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[а-яa-z0-9]+", text)
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


def document_frequency(tokens_list: Iterable[List[str]]) -> Counter:
    cnt: Counter = Counter()
    for toks in tokens_list:
        cnt.update(set(toks))
    return cnt


def top_lift_terms(
    tokens_list: List[List[str]],
    high_mask: pd.Series,
    min_high_docs: int = 20,
    top_n: int = 20,
) -> pd.DataFrame:
    high_tokens = [tokens_list[i] for i, is_high in enumerate(high_mask.tolist()) if is_high]
    rest_tokens = [tokens_list[i] for i, is_high in enumerate(high_mask.tolist()) if not is_high]

    high_df = document_frequency(high_tokens)
    rest_df = document_frequency(rest_tokens)

    n_high = len(high_tokens)
    n_rest = len(rest_tokens)

    rows = []
    for token, h_count in high_df.items():
        if h_count < min_high_docs:
            continue
        r_count = rest_df.get(token, 0)
        p_high = h_count / max(n_high, 1)
        p_rest = r_count / max(n_rest, 1)
        lift = p_high / max(p_rest, 1e-6)
        rows.append(
            {
                "token": token,
                "high_doc_count": h_count,
                "rest_doc_count": r_count,
                "high_share": round(p_high, 4),
                "rest_share": round(p_rest, 4),
                "lift": round(lift, 3),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["token", "high_doc_count", "rest_doc_count", "high_share", "rest_share", "lift"])

    out = pd.DataFrame(rows).sort_values(["lift", "high_doc_count"], ascending=[False, False]).head(top_n)
    return out.reset_index(drop=True)


def compute_behavior_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    for metric in BEHAVIOR_WEIGHTS:
        scored[f"behavior_{metric}"] = pct_rank(scored[metric], higher_is_better=BEHAVIOR_DIRECTION[metric])

    scored["behavior_score_raw"] = 0.0
    for metric, w in BEHAVIOR_WEIGHTS.items():
        scored["behavior_score_raw"] += scored[f"behavior_{metric}"] * w
    scored["behavior_score"] = (scored["behavior_score_raw"] * 100).round(2)
    return scored


def compute_style_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["text_title"] = normalize_text(out["ad_title_example"])
    out["text_desc"] = normalize_text(out["ad_description_example"])
    out["text_full"] = (out["text_title"] + " " + out["text_desc"]).str.strip()

    out["text_word_count"] = out["text_full"].str.replace(r"[^а-яa-z0-9\s]", " ", regex=True).str.split().str.len().fillna(0)
    out["text_digit_count"] = out["text_full"].str.count(r"\d")
    out["text_unique_ratio"] = out["text_full"].str.replace(r"[^а-яa-z0-9\s]", " ", regex=True).str.split().apply(
        lambda x: len(set(x)) / len(x) if len(x) > 0 else 0
    )

    out["kw_professional_count"] = count_terms(out["text_full"], PROFESSIONAL_TERMS)
    out["kw_logistics_count"] = count_terms(out["text_full"], LOGISTICS_TERMS)
    out["kw_charisma_count"] = count_terms(out["text_full"], CHARISMA_TERMS)
    out["kw_expertise_count"] = count_terms(out["text_full"], EXPERTISE_TERMS)

    # Professionalism marker
    out["score_professionalism"] = (
        0.40 * pct_rank(out["kw_professional_count"], True)
        + 0.20 * pct_rank(out["kw_logistics_count"], True)
        + 0.20 * pct_rank(out["rating_seller"].fillna(out["rating_seller"].median()), True)
        + 0.20 * pct_rank(out["response_time_minutes"].fillna(out["response_time_minutes"].median()), False)
    )

    # Charisma marker
    out["score_charisma"] = (
        0.35 * pct_rank(out["kw_charisma_count"], True)
        + 0.25 * pct_rank(out["comments_per_ad_avg"], True)
        + 0.20 * pct_rank(out["comment_sentiment_avg"], True)
        + 0.10 * pct_rank(out["comment_length_avg"], True)
        + 0.10 * pct_rank(out["followers_count"], True)
    )

    # Expertise marker
    out["score_expertise"] = (
        0.35 * pct_rank(out["kw_expertise_count"], True)
        + 0.20 * pct_rank(out["questions_per_ad_avg"].fillna(out["questions_per_ad_avg"].median()), True)
        + 0.20 * pct_rank(out["deal_success_rate"], True)
        + 0.15 * pct_rank(out["text_digit_count"], True)
        + 0.10 * pct_rank(out["text_word_count"], True)
    )

    out["style_score_raw"] = (
        0.40 * out["score_professionalism"]
        + 0.30 * out["score_charisma"]
        + 0.30 * out["score_expertise"]
    )
    out["style_score"] = (out["style_score_raw"] * 100).round(2)

    return out


def rank_with_blend(df: pd.DataFrame, style_share: float) -> pd.DataFrame:
    behavior_share = 1.0 - style_share
    ranked = df.copy()
    ranked["final_score_raw"] = behavior_share * ranked["behavior_score_raw"] + style_share * ranked["style_score_raw"]
    ranked["final_score"] = (ranked["final_score_raw"] * 100).round(2)
    ranked = ranked.sort_values("final_score", ascending=False).copy()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def top_ids(df: pd.DataFrame, k: int = TOP_K) -> Set[str]:
    return set(df.head(min(k, len(df)))["user_id"].tolist())


def overlap_stats(base_ids: Set[str], other_ids: Set[str]) -> dict:
    inter = len(base_ids & other_ids)
    union = len(base_ids | other_ids)
    return {
        "overlap_count": inter,
        "overlap_share_of_base": round(inter / max(len(base_ids), 1), 4),
        "jaccard": round(inter / max(union, 1), 4),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_data(DATA_PATH)
    cleaned, clean_stats = clean_users(raw)
    sellers = cleaned[cleaned["is_seller"] == 1].copy()

    eligible = apply_eligibility(sellers, BASE_ELIGIBILITY)
    eligible = compute_behavior_score(eligible)
    eligible = compute_style_scores(eligible)

    # Base blend for step6
    base_style_share = 0.30
    ranked_base = rank_with_blend(eligible, style_share=base_style_share)
    top1000_step6 = ranked_base.head(min(TOP_K, len(ranked_base))).copy()

    # Save enriched top1000
    keep_cols = [
        "rank",
        "user_id",
        "final_score",
        "behavior_score",
        "style_score",
        "score_professionalism",
        "score_charisma",
        "score_expertise",
        "kw_professional_count",
        "kw_logistics_count",
        "kw_charisma_count",
        "kw_expertise_count",
        "text_word_count",
        "text_digit_count",
        "text_unique_ratio",
        "user_segment_marketing",
        "user_city",
        "user_region",
        "ad_category_1",
        "ad_category_2",
        "ad_category_3",
        "rating_seller",
        "reviews_count",
        "response_time_minutes",
        "comments_per_ad_avg",
        "comment_length_avg",
        "comment_sentiment_avg",
        "questions_per_ad_avg",
        "followers_count",
        "total_ads_last_12m",
        "total_deals_last_12m",
        "deal_success_rate",
    ]
    top1000_step6[keep_cols].to_csv(OUT_DIR / "top1000_candidates_step6_text.csv", index=False)

    # Full eligible score table (for audits)
    eligible[
        [
            "user_id",
            "behavior_score",
            "style_score",
            "score_professionalism",
            "score_charisma",
            "score_expertise",
            "kw_professional_count",
            "kw_logistics_count",
            "kw_charisma_count",
            "kw_expertise_count",
            "text_word_count",
            "text_digit_count",
            "text_unique_ratio",
            "ad_category_1",
            "user_segment_marketing",
        ]
    ].to_csv(OUT_DIR / "eligible_scores_step6.csv", index=False)

    # Sensitivity by blend weight
    step5_base = pd.read_csv(STEP5_TOP1000_PATH, dtype={"user_id": "string"})
    step5_ids = set(step5_base["user_id"].tolist())

    blend_rows = []
    scenario_ids: Dict[str, Set[str]] = {}
    for style_share in [0.20, 0.30, 0.40]:
        scenario_name = f"style_{int(style_share*100)}"
        ranked = rank_with_blend(eligible, style_share=style_share)
        ids = top_ids(ranked)
        scenario_ids[scenario_name] = ids
        row = {
            "scenario": scenario_name,
            "style_share": style_share,
            "behavior_share": round(1 - style_share, 2),
            "eligible_users": len(eligible),
            "top_k_used": min(TOP_K, len(ranked)),
        }
        row.update({f"vs_step5_{k}": v for k, v in overlap_stats(step5_ids, ids).items()})
        blend_rows.append(row)

    # Pairwise overlap between style scenarios
    base_ids = scenario_ids["style_30"]
    for row in blend_rows:
        sid = scenario_ids[row["scenario"]]
        row.update({f"vs_style30_{k}": v for k, v in overlap_stats(base_ids, sid).items()})

    blend_sens = pd.DataFrame(blend_rows)
    blend_sens.to_csv(OUT_DIR / "blend_sensitivity.csv", index=False)

    # Marker summary for top1000
    marker_summary = pd.DataFrame(
        {
            "metric": [
                "score_professionalism",
                "score_charisma",
                "score_expertise",
                "kw_professional_count",
                "kw_logistics_count",
                "kw_charisma_count",
                "kw_expertise_count",
                "text_word_count",
                "text_digit_count",
            ],
            "eligible_mean": [
                eligible["score_professionalism"].mean(),
                eligible["score_charisma"].mean(),
                eligible["score_expertise"].mean(),
                eligible["kw_professional_count"].mean(),
                eligible["kw_logistics_count"].mean(),
                eligible["kw_charisma_count"].mean(),
                eligible["kw_expertise_count"].mean(),
                eligible["text_word_count"].mean(),
                eligible["text_digit_count"].mean(),
            ],
            "top1000_mean": [
                top1000_step6["score_professionalism"].mean(),
                top1000_step6["score_charisma"].mean(),
                top1000_step6["score_expertise"].mean(),
                top1000_step6["kw_professional_count"].mean(),
                top1000_step6["kw_logistics_count"].mean(),
                top1000_step6["kw_charisma_count"].mean(),
                top1000_step6["kw_expertise_count"].mean(),
                top1000_step6["text_word_count"].mean(),
                top1000_step6["text_digit_count"].mean(),
            ],
        }
    )
    marker_summary["lift_top1000_vs_eligible"] = (marker_summary["top1000_mean"] / marker_summary["eligible_mean"]).round(3)
    marker_summary.to_csv(OUT_DIR / "marker_summary.csv", index=False)

    # Distribution by categories for marker-driven interpretation
    category_marker = (
        top1000_step6.groupby("ad_category_1")
        .agg(
            users=("user_id", "count"),
            final_score_mean=("final_score", "mean"),
            professionalism_mean=("score_professionalism", "mean"),
            charisma_mean=("score_charisma", "mean"),
            expertise_mean=("score_expertise", "mean"),
        )
        .sort_values("users", ascending=False)
        .reset_index()
    )
    category_marker.to_csv(OUT_DIR / "top1000_category_marker_scores.csv", index=False)

    # Stylized term extraction (top decile for each marker)
    tokens_list = [tokenize(t) for t in eligible["text_full"].tolist()]
    high_prof = eligible["score_professionalism"] >= eligible["score_professionalism"].quantile(0.9)
    high_char = eligible["score_charisma"] >= eligible["score_charisma"].quantile(0.9)
    high_exp = eligible["score_expertise"] >= eligible["score_expertise"].quantile(0.9)

    top_lift_terms(tokens_list, high_prof, min_high_docs=30, top_n=25).to_csv(
        OUT_DIR / "top_terms_professionalism.csv", index=False
    )
    top_lift_terms(tokens_list, high_char, min_high_docs=30, top_n=25).to_csv(
        OUT_DIR / "top_terms_charisma.csv", index=False
    )
    top_lift_terms(tokens_list, high_exp, min_high_docs=30, top_n=25).to_csv(
        OUT_DIR / "top_terms_expertise.csv", index=False
    )

    # Summary markdown
    top1 = top1000_step6.iloc[0]
    base_overlap = overlap_stats(step5_ids, set(top1000_step6["user_id"]))

    summary_lines = [
        "# Шаг 6: NLP-маркеры стиля + обновленный ranking",
        "",
        "## 1) Facts from data",
        f"- После очистки: {clean_stats['rows_after_dedup']:,} пользователей; sellers={len(sellers):,}; eligible pool={len(eligible):,}.",
        f"- Базовый blend: behavior 70% + style 30%.",
        f"- Сформирован `top-{min(TOP_K, len(top1000_step6))}`: `analysis/step6_outputs/top1000_candidates_step6_text.csv`.",
        "",
        "## 2) NLP/Style markers",
        "### Professionalism",
        "- Сигналы: quality/transaction keywords + seller rating + скорость ответа.",
        "### Charisma",
        "- Сигналы: CTA keywords + комментарии (объем/тональность/длина) + followers.",
        "### Expertise",
        "- Сигналы: domain keywords + questions_per_ad + deal_success_rate + текстовая специфичность (цифры/длина).",
        "",
        "## 3) Ranking snapshot",
        f"- Top-1: user_id={top1['user_id']}, final_score={top1['final_score']:.2f}, behavior_score={top1['behavior_score']:.2f}, style_score={top1['style_score']:.2f}.",
        f"- Overlap с step5 behavior-only top1000: {base_overlap['overlap_count']} ({base_overlap['overlap_share_of_base']*100:.2f}%), jaccard={base_overlap['jaccard']:.4f}.",
        "",
        "## 4) Sensitivity (blend weight)",
    ]

    for _, row in blend_sens.iterrows():
        summary_lines.append(
            f"- {row['scenario']}: overlap vs step5 = {int(row['vs_step5_overlap_count'])} ({row['vs_step5_overlap_share_of_base']*100:.2f}%), jaccard={row['vs_step5_jaccard']:.4f}."
        )

    summary_lines.extend(
        [
            "",
            "## 5) Interpretations (not facts)",
            "- NLP-маркеры меняют состав top-листа умеренно, сохраняя большую часть поведенческого ядра.",
            "- На synthetic-тексте словари работают как explainable layer, но для production-нужна валидация на реальных UGC-текстах.",
            "",
            "## 6) ASSUMPTIONS",
            "- ASSUMPTION: агрегированные поля комментариев (`comment_*`) можно использовать как прокси харизмы при отсутствии сырых комментариев.",
            "- ASSUMPTION: словари ключевых фраз достаточны для MVP-ранжирования и презентации, хотя не заменяют full NLP-модель.",
            "",
            "## Артефакты",
            "- `analysis/step6_outputs/top1000_candidates_step6_text.csv`",
            "- `analysis/step6_outputs/eligible_scores_step6.csv`",
            "- `analysis/step6_outputs/blend_sensitivity.csv`",
            "- `analysis/step6_outputs/marker_summary.csv`",
            "- `analysis/step6_outputs/top1000_category_marker_scores.csv`",
            "- `analysis/step6_outputs/top_terms_professionalism.csv`",
            "- `analysis/step6_outputs/top_terms_charisma.csv`",
            "- `analysis/step6_outputs/top_terms_expertise.csv`",
        ]
    )

    SUMMARY_MD.write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
