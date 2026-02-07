# Краткие выводы ресерча (Avito Лента)

## 1) Краткий вывод
На данных сформировано узкое, но качественное “золотое ядро” про-пользователей. Для первой волны запуска оптимальный сегмент — **Электроника**; fallback-сегмент — **Авто**.

Выбор устойчив по сценариям (strict/relaxed/step5/step6) и подтверждается балансом масштаба и вовлеченности.

## 2) Факты из данных
- После очистки: **89,982** пользователей; sellers: **49,547**.
- Размер про-сегмента по правилам:
  - `strict`: **193** (0.214% от всей базы; 0.390% от sellers)
  - `relaxed`: **881** (0.979% от всей базы; 1.778% от sellers)
  - `step6 top-1000`: **1000** (1.111% от всей базы; 2.018% от sellers)
- Категории из условия кейса в `step6 top-1000`:
  - **Электроника**: users=226, concentration_idx=1.183, engagement_idx=1.191, priority_score=1.409
  - **Авто**: users=193, concentration_idx=1.205, engagement_idx=1.157, priority_score=1.394
  - Недвижимость: users=103, concentration_idx=0.941, engagement_idx=1.282, priority_score=1.206
  - Хобби и отдых: users=92, concentration_idx=0.771, engagement_idx=1.261, priority_score=0.972
- Robustness: **Электроника** — топ-1 в **3 из 4** сценариев, средний ранг **1.25**.

## 3) Интерпретации
- Электроника дает лучший компромисс между размером сегмента и потенциальной вовлеченностью.
- Авто близко по приоритету и подходит как операционный fallback на пилот.
- NLP-слой (step6) не ломает поведенческое ядро, но заметно улучшает качество отбора для контентного сценария.

## 4) Почему именно Электроника (vs seller-base)
- `total_ads_last_12m`: x3.125
- `total_deals_last_12m`: x3.236
- `questions_per_ad_avg`: x1.813
- `reviews_count`: x3.107
- `deal_success_rate`: x1.073
- `response_time_minutes`: быстрее в x1.442

## 5) Финальные решения и артефакты
- Pilot segment: **Электроника**
- Fallback segment: **Авто**
- Short-list для outreach:
  - `analysis/final_outputs/shortlist_top200_electronics_step6_with_tiers.csv` (200)
  - `analysis/final_outputs/shortlist_top200_auto_step6_with_tiers.csv` (193)

## 6) Риски и ограничения
- ASSUMPTION: `priority_score = concentration_idx * engagement_idx` корректно отражает пилотный потенциал.
- ASSUMPTION: словарные NLP-маркеры и агрегаты комментариев достаточны для MVP-ранжирования.
- Fallback-список Авто ограничен 193 кандидатами в пределах текущего `step6 top-1000`.
