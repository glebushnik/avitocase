# Датасет-блок: 2 слайда

## Слайд 1 — Как выбрали про-сегмент
**Заголовок:** От полной базы к пилотному про-сегменту

**Что показать на слайде (контент):**
- База после очистки: **89,982** пользователей, из них sellers: **49,547**.
- Воронка определений про-сегмента:
  - `strict`: **193** (0.214% от всех; 0.390% от sellers)
  - `relaxed`: **881** (0.979% от всех; 1.778% от sellers)
  - `step6 top-1000`: **1000** (1.111% от всех; 2.018% от sellers)
- Критерий выбора категории для пилота:
  - `concentration_idx` = доля категории в про-сегменте / доля категории у sellers
  - `engagement_idx` = композит вовлеченности
  - `priority_score = concentration_idx * engagement_idx`
- Результат в `step6 top-1000` (категории из кейса):
  - Электроника: users=226, concentration=1.183, engagement=1.191, priority=1.409
  - Авто: users=193, concentration=1.205, engagement=1.157, priority=1.394
  - Недвижимость: users=103, concentration=0.941, engagement=1.282, priority=1.206
  - Хобби и отдых: users=92, concentration=0.771, engagement=1.261, priority=0.972
- Robustness-check по 4 сценариям: Электроника — топ-1 в **3 из 4** сценариев, средний ранг **1.25**.

**Визуал:**
- Слева: funnel `strict -> relaxed -> step6 top-1000`.
- Справа: grouped bar по `priority_score` для 4 категорий (Авто/Недвижимость/Электроника/Хобби).

**So what (1 строка):**
- Электроника дает лучший баланс масштаба и вовлеченности и выигрывает по устойчивости выбора.

---

## Слайд 2 — Почему именно Электроника для пилота
**Заголовок:** Электроника как стартовый сегмент + Авто как fallback

**Что показать на слайде (контент):**
- Финальное решение:
  - Pilot segment: **Электроника**
  - Fallback segment: **Авто**
- Почему Электроника (vs seller-base):
  - `total_ads_last_12m`: x3.125
  - `total_deals_last_12m`: x3.236
  - `questions_per_ad_avg`: x1.813
  - `reviews_count`: x3.107
  - `deal_success_rate`: x1.073
  - `response_time_minutes`: скорость ответа лучше в x1.442
- Outreach short-list из `top1000_candidates_step6_text.csv`:
  - Primary (`Электроника`): **200** кандидатов (`shortlist_top200_electronics_step6_with_tiers.csv`)
  - Fallback (`Авто`): **193** кандидата (`shortlist_top200_auto_step6_with_tiers.csv`)
  - Tiering для запуска outreach:
    - T1 priority: 50
    - T2 core: 70
    - T3 reserve: 80 (для fallback: 73)

**Визуал:**
- Слева: KPI-card “Почему Электроника” (6 лифтов).
- Справа: stacked bars по outreach-тирами для primary/fallback.

**So what (1 строка):**
- Пилот в Электронике максимизирует шанс быстрого сетевого эффекта, а Авто покрывает риск сценария с альтернативной категорией.

## Файлы для презентации
- `analysis/final_outputs/slide1_step6_case_categories.csv`
- `analysis/final_outputs/slide2_electronics_vs_auto_priority.csv`
- `analysis/final_outputs/slide2_electronics_profile_key_metrics.csv`
- `analysis/final_outputs/shortlist_top200_electronics_step6_with_tiers.csv`
- `analysis/final_outputs/shortlist_top200_auto_step6_with_tiers.csv`
- `analysis/final_outputs/pilot_segment_config.json`
