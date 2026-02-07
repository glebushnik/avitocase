# Шаги 2-4: Data Audit + EDA + Baseline Pro-user

## 1) Facts from data (Step 2: Data understanding & quality)
- Размер: **90,000** строк, **36** колонок, уникальных `user_id`: **89,999**.
- Период регистраций: **2018-01-01 — 2023-09-30**.
- Период последней активности: **2022-10-31 — 2023-10-31**.
- Валидация `user_id`: валидных hex-16 id **89,982**, невалидных id **18** (уникальных невалидных: **17**).
- Дубликаты: по полным строкам **0**, по `user_id` **1**.

### Пропуски (топ)
- `response_time_minutes`: **49.29%**
- `positive_reviews_percent`: **47.61%**
- `rating_seller`: **47.07%**
- `photo_quality_score`: **38.14%**
- `avg_ad_lifetime_days`: **5.95%**
- `questions_per_ad_avg`: **5.06%**

### Логические проверки
- `active_ads_current > total_ads_last_12m`: **0**.
- `deal_success_rate` вне [0,1]: **0**.
- `positive_reviews_percent` вне [0,100]: **0**.
- `last_activity_date < registration_date`: **813**.
- `is_seller=0` и `total_ads_last_12m>0`: **18276**.
- `is_buyer=0` и `total_deals_last_12m>0`: **19543**.

## 2) EDA (Step 3)
- Ролевой микс: sellers **55.06%**, buyers **74.83%**, обе роли **41.16%**.
- Только sellers: **13.90%**, только buyers: **33.67%**, ни одна роль: **11.26%**.

### Основные хвосты распределений
- `total_ads_last_12m`: p50=9.00, p90=76.00, p99=248.00, max=402.00.
- `total_revenue_last_12m`: p50=26011.50, p90=1079588.10, p99=34593912.44, max=200000000.00.
- `questions_per_ad_avg`: p50=1.40, p90=4.30, p99=8.36, max=22.00.
- `followers_count`: p50=181.00, p90=748.00, p99=2299.00, max=16419.00.

### Крупнейшие категории (ad_category_1)
- Электроника: **17,021**
- Авто: **14,278**
- Услуги: **12,622**
- Хобби и отдых: **10,828**
- Недвижимость: **9,849**
- Дом и сад: **9,099**
- Одежда и обувь: **9,088**
- Дети и игрушки: **7,215**

## 3) Baseline сегментации 'Про-пользователь' (Step 4)
### Strict baseline rule
- `is_seller=1` AND `total_ads_last_12m >= 61.0` AND `avg_ad_lifetime_days <= 3.3` AND `questions_per_ad_avg >= 3.5` AND `rating_seller >= 4.9` AND `reviews_count >= 43.0`.
- Результат strict: **193** пользователей (**0.21%** от всей базы, **0.39%** от sellers).

### Relaxed baseline rule (для целевого пула ~1000 на следующем шаге)
- `is_seller=1` AND `total_ads_last_12m >= 46.0` AND `avg_ad_lifetime_days <= 4.0` AND `questions_per_ad_avg >= 2.8` AND `rating_seller >= 4.8` AND `reviews_count >= 10`.
- Результат relaxed: **881** пользователей (**0.98%** от всей базы, **1.78%** от sellers).

## 4) Interpretations (not facts)
- Большая доля пропусков в `rating_seller`, `positive_reviews_percent`, `response_time_minutes` в основном структурная: метрики не применяются к части non-sellers.
- Маркетинговые сегменты (`Новичок`, `Активный`, `Премиум` и т.д.) в текущем срезе слабо разделяют пользователей по core-операционным метрикам, поэтому для отбора кандидатов лучше опираться на поведенческие признаки.
- Распределения выручки/объемов сделок сильно скошены вправо; медианные и перцентильные пороги надежнее средних для правил сегментации.

## 5) ASSUMPTIONS
- ASSUMPTION: `is_seller`/`is_buyer` отражают текущую роль пользователя, а исторические активности (`total_ads_last_12m`, `total_deals_last_12m`) могут включать прошлые роли; поэтому комбинации вида `is_seller=0` + `total_ads_last_12m>0` считаем не ошибкой по умолчанию.
- ASSUMPTION: для baseline сегментации скорость обновления объявлений приближенно измеряется как низкий `avg_ad_lifetime_days`.

## Артефакты
- `analysis/step2_4_outputs/missingness.csv`
- `analysis/step2_4_outputs/logic_checks.csv`
- `analysis/step2_4_outputs/quantiles.csv`
- `analysis/step2_4_outputs/segment_metrics.csv`
- `analysis/step2_4_outputs/category_focus_metrics.csv`
- `analysis/step2_4_outputs/pro_user_rule_summary.csv`
- `analysis/step2_4_outputs/pro_user_overindex.csv`