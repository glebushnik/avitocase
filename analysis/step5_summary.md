# Шаг 5: Scoring + Ranking Top-1000 кандидатов

## 1) Facts from data
- После очистки ID и дедупликации: 89,982 строк (удалено невалидных ID: 18, удалено дублей ID: 0).
- Sellers после очистки: 49,547.
- Eligible pool (base thresholds): 39,441.
- Сформирован рейтинг top-1000 кандидатов: `analysis/step5_outputs/top1000_candidates.csv`.

## 2) Scoring rule (transparent)
- `ProScore` = взвешенная сумма перцентильных component-скоров (0..1), далее масштаб в 0..100.
- Перцентильные скоры считаются только по sellers; для missing у компонента используется нейтральное значение 0.5.

### Метрики и веса
- `total_ads_last_12m`: weight=0.20, higher_is_better
- `avg_ad_lifetime_days`: weight=0.13, lower_is_better
- `questions_per_ad_avg`: weight=0.17, higher_is_better
- `rating_seller`: weight=0.14, higher_is_better
- `reviews_count`: weight=0.10, higher_is_better
- `response_time_minutes`: weight=0.08, lower_is_better
- `total_deals_last_12m`: weight=0.08, higher_is_better
- `deal_success_rate`: weight=0.06, higher_is_better
- `contacts_per_ad_avg`: weight=0.04, higher_is_better

## 3) Base eligibility rule
- `is_seller=1` AND `total_ads_last_12m >= 10` AND `reviews_count >= 5` AND `questions_per_ad_avg >= 0.5`.

## 4) Ranking snapshot
- Top-1 ProScore: 93.51 (user_id=f87d024b1e31a3eb).
- Минимальный ProScore в top-10: 88.26.
- Медиана ProScore в top-1000: 77.65.
- Минимальный ProScore в top-1000: 75.08.

## 5) Sensitivity checks
### Threshold sensitivity (same weights)
- lenient: eligible=44906, overlap_with_base=975 (97.50% от base top-1000), jaccard=0.9512.
- base: eligible=39441, overlap_with_base=1000 (100.00% от base top-1000), jaccard=1.0000.
- strict: eligible=25851, overlap_with_base=916 (91.60% от base top-1000), jaccard=0.8450.

### Weight sensitivity (same eligibility)
- balanced_base: overlap_with_base=1000 (100.00% от base top-1000), jaccard=1.0000.
- engagement_heavy: overlap_with_base=695 (69.50% от base top-1000), jaccard=0.5326.
- trust_heavy: overlap_with_base=784 (78.40% от base top-1000), jaccard=0.6447.

## 6) Interpretations (not facts)
- Если overlap высок даже при изменении весов/порогов, ядро кандидатов устойчиво и подходит для пилотного набора.
- Если overlap заметно падает при strict thresholds, значит долгосрочно понадобится расширенная воронка (вторая волна кандидатов).

## 7) ASSUMPTIONS
- ASSUMPTION: neutral fill (0.5) для missing component-скора приемлем, т.к. внутри sellers доля пропусков по ключевым метрикам относительно низкая.
- ASSUMPTION: выбранные веса отражают продуктовый приоритет пилота (активность + вовлечение + доверие), а не долгосрочную монетизацию.

## Артефакты
- `analysis/step5_outputs/top1000_candidates.csv`
- `analysis/step5_outputs/threshold_sensitivity.csv`
- `analysis/step5_outputs/weight_sensitivity.csv`
- `analysis/step5_outputs/top1000_category_distribution.csv`
- `analysis/step5_outputs/top1000_segment_distribution.csv`