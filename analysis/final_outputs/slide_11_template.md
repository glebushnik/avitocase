# Слайд 11 — Датасет: категории для старта + топ‑1000 кандидатов + roadmap

## Заголовок слайда
**Почему стартуем с Электроники и как запускаем pilot/fallback**

## Подзаголовок
Сравнение категорий + финальный shortlist + план запуска

## Контент на слайд

### 1) Категории для старта (из step6 top-1000)
- **Электроника**: users=**226**, concentration_idx=**1.183**, engagement_idx=**1.191**, priority_score=**1.409**
- **Авто**: users=**193**, concentration_idx=**1.205**, engagement_idx=**1.157**, priority_score=**1.394**
- Недвижимость: users=103, priority_score=1.206
- Хобби и отдых: users=92, priority_score=0.972

**Решение:**
- Pilot segment: **Электроника**
- Fallback segment: **Авто**

### 2) Финальный пул кандидатов
- Общий пул: `top1000_candidates_step6_text.csv` (1,000 пользователей)
- Outreach shortlist (primary):  
  - `shortlist_top200_electronics_step6_with_tiers.csv` → **200**
- Outreach shortlist (fallback):  
  - `shortlist_top200_auto_step6_with_tiers.csv` → **193**

### 3) Tiering для outreach
- `T1_priority`: 50
- `T2_core`: 70
- `T3_reserve`: 80 (для fallback: 73)

### 4) Roadmap запуска (предложение)
- **Фаза 1 (Неделя 1–2):** Outreach по `T1_priority` в Электронике, сбор первых откликов
- **Фаза 2 (Неделя 3–4):** Подключение `T2_core`, первичные интеграции и замер KPI
- **Фаза 3 (Месяц 2):** Подключение `T3_reserve`, оптимизация воронки
- **Фаза 4 (Месяц 2–3):** Если KPI не достигаются в Электронике — параллельный rollout в Авто (fallback)

## So what (1 строка)
Электроника дает лучший старт по качеству ядра и масштабу, а Авто снижает риск недобора KPI в пилоте.

## Рекомендованный визуал
- Слева: bar chart `priority_score` (Электроника/Авто/Недвижимость/Хобби)
- Справа: timeline roadmap + mini-table по shortlist (200 / 193)

## Source data
- `analysis/final_outputs/slide1_step6_case_categories.csv`
- `analysis/final_outputs/slide2_electronics_vs_auto_priority.csv`
- `analysis/final_outputs/shortlist_outreach_summary.csv`
- `analysis/final_outputs/pilot_segment_config.json`
